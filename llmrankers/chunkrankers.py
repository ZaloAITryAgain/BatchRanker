from collections import defaultdict
import time
from tracemalloc import start
from typing import Dict, List
import statistics
import tiktoken
import asyncio
import random
import os
from openai import AsyncOpenAI

from llmrankers.utils.prompt import (
    SetwiseRankRelevancePrompt,
    BatchwiseRankRelevancePrompt,
    ChunkwiseRankRelevancePrompt,
)
from llmrankers.utils.llm_client import LLMClient
from llmrankers.utils.logging import logger
from llmrankers.utils.llm_schema import (
    TrialScoringNoCOTBatchResponse,
    TrialScoringBatchResponse,
    BatchScoringResponseType,
    MultiGenerationsResponse,
    Model,
)
from llmrankers.utils.prompt import SearchResult
from abc import abstractmethod, ABC
from llmrankers.utils.llm_schema import (
    DocumentRelevanceScoresBatchResponse,
    DocumentRelevanceScoresWithReasonBatchResponse,
)
from llmrankers.utils.llm_schema import (
    LiteLLMKwargs,
    MultiGenerationsResponse,
    Provider,
    ChatMessage,
    MessageRole,
    ParseType,
)
from pydantic import BaseModel

from transformers import AutoTokenizer


async def stop():
    loop = asyncio.get_event_loop()
    loop.stop()
    loop.close()


def traceback_wrapper(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            import traceback

            traceback.print_exc()
            return {}

    return wrapper


class Anchor(BaseModel):
    docid: str
    text: str
    score: float
    reasoning: str = ""


class BaseChunkwiseRanker(ABC):
    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 10,
        num_vote: int = 5,
        method: str = "random",
        temperature: float = 0.5,
        num_anchor: int = 5,
        use_COT: bool = False,
        use_COT_anchor: bool = False,
    ):
        self.batch_size = batch_size
        self.num_vote = num_vote
        self.method = method
        self.llm = model_name_or_path
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.temperature = temperature
        self.num_anchor = num_anchor
        self.use_COT_document = use_COT
        self.use_COT_anchor = use_COT_anchor

        self.print_example = False
        self.delay_per_query = 0

        self.client = None
        self.llm_client = None
        self.extra_time_for_asyncio = 0

    def rerank(
        self,
        query: str,
        ranking: List[SearchResult],
    ) -> List[SearchResult]:
        # reset the total counts every time we rerank
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

        # if not self.use_COT_document:
        time.sleep(self.delay_per_query)
        # result = asyncio.run(self._rerank(query, ranking))
        # if not self.client:
        #     self.client.close()
        # if not self.llm_client:
        #     self.llm_client.close()
        # return result

        try:
            # Get the existing event loop if there is one, or create a new one
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run the async operation
            result = loop.run_until_complete(self._rerank(query, ranking))

            return result

        except Exception as e:
            logger.error(f"Error in rerank: {e}")
            raise e
        finally:
            start_asyncio_time = time.time()
            # Don't close the loop, just clean up the task
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()

            # Wait until all tasks are cancelled
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self.extra_time_for_asyncio += time.time() - start_asyncio_time

    async def _rerank(
        self,
        query: str,
        ranking: List[SearchResult],
    ) -> List[SearchResult]:
        """
        perform batch relevance scoring in two steps:
        1. Select the top trial from each batch. These act as anchors and the trials are scored relative to them.
        2. Perform self-consistency COT batching on all batches:
            - The prompt takes into account the top trials selected in the previous step.
            - Mathematical validation:
                TrialScore[i] = Pr(Trial_i is most relevant | batch_1, batch_2, ..., batch_n trials)
                              ~ Pr(Trial_i is most relevant | batch_1_anchor, batch_2_anchor, ..., batch_n_anchor)

              The above equality holds because:
                Pr(X > Y & X > Z | Y > Z) = Pr(X > Y | Y > Z)

              Assumption:
                - The LLM has enough "depth" to reason about a batch of trials and select the most relevant one.
                    * This could be minimized using self-consistency on Chain of Thought (CoT) batching.
        """
        # Step 1: Select the top trial from each batch. These act as anchors and the trials are scored relative to them.
        # Currently select the trials with highest similarity score as anchors. TODO: implement a proper top trial selection method.
        if self.method == "random":
            top_docs = random.sample(ranking, k=self.num_anchor)
        elif self.method == "top":
            top_docs = sorted(ranking, key=lambda x: x.score, reverse=True)[: self.num_anchor]
        elif self.method == "none":
            top_docs = []
        else:
            raise NotImplementedError

        remaining_docs = [doc for doc in ranking if doc not in top_docs]

        # Step 2: Get scores of the top docs/anchors
        anchors = await self._rank_anchors(query=query, anchors=top_docs)
        self.total_compare += 1
        print(f"Done ranking anchors")

        # Step 3: Perform self-consistency COT batching on all batches
        batches = [
            remaining_docs[i : i + self.batch_size]
            for i in range(0, len(remaining_docs), self.batch_size)
        ]
        all_scoring_process = []
        for _, batch in enumerate(batches):
            _scoring_process = self.get_score_batch(
                query=query,
                batch=batch,
                anchors=anchors,
            )
            all_scoring_process.append(_scoring_process)
        batched_scores = await asyncio.gather(*[score for score in all_scoring_process])
        self.total_compare += len(all_scoring_process)
        print(f"Done scoring batches")

        # Step 3: Aggregate the results from all batches and return the final ranking.
        aggregated_scores = defaultdict(list)
        for score_dict in batched_scores:
            for doc_id, score_list in score_dict.items():
                aggregated_scores[doc_id].extend(score_list)
        aggregated_scores = {
            doc_id: statistics.median(scores) for doc_id, scores in aggregated_scores.items()
        }
        aggregated_scores.update({doc.docid: doc.score for doc in anchors})
        final_ranking = [
            SearchResult(docid=doc_id, score=score, text="")
            for doc_id, score in aggregated_scores.items()
        ]
        print(f"Done aggregating scores")
        if len(final_ranking) != len(ranking):
            all_ranking_ids = {doc.docid for doc in ranking}
            final_ranking_ids = {doc.docid for doc in final_ranking}
            missing_docs = all_ranking_ids - final_ranking_ids
            extra_docs = final_ranking_ids - all_ranking_ids
            if missing_docs:
                print(f"Missing {len(missing_docs)} doc: {missing_docs}")
                final_ranking.extend(
                    [SearchResult(docid=docid, score=0, text="") for docid in missing_docs]
                )
            if extra_docs:
                print(f"Extra {len(extra_docs)} doc: {extra_docs}")
                final_ranking = [doc for doc in final_ranking if doc.docid not in extra_docs]

        sorted_docs = sorted(final_ranking, key=lambda x: x.score, reverse=True)[: len(ranking)]
        return sorted_docs

    @traceback_wrapper
    async def _rank_anchors(
        self,
        query: str,
        anchors: list[SearchResult],
    ):
        get_score_prompt = ChunkwiseRankRelevancePrompt(
            query=query,
            anchors=None,
            documents=anchors,
            use_COT=self.use_COT_anchor,
            num_anchor=self.num_anchor,
            output_schema=(
                DocumentRelevanceScoresBatchResponse
                if not self.use_COT_anchor
                else DocumentRelevanceScoresWithReasonBatchResponse
            ),
        )

        score_anchors_response = await self.ask_llm(get_score_prompt)

        # make sure this works even if num_vote is 1, which causes the LLM to generate only one response, not a list
        if not isinstance(score_anchors_response, MultiGenerationsResponse):
            score_anchors_response = MultiGenerationsResponse(results=[score_anchors_response])
        self.total_completion_tokens += score_anchors_response.completion_tokens
        self.total_prompt_tokens += score_anchors_response.prompt_tokens

        # gather the results from all generations
        gathered_scores = defaultdict(list)
        key_to_document = get_score_prompt.key_to_document
        results: list = score_anchors_response.results
        for score_path in results:
            for doc_data in score_path.document_scores:
                document_id = doc_data.document_id
                relevance_score = doc_data.relevance_score
                if document_id not in key_to_document:
                    continue
                document_original_id = key_to_document[document_id]
                gathered_scores[document_original_id].append(relevance_score)

        anchor_scores = {i: statistics.median(scores) for i, scores in gathered_scores.items()}

        # For reasonings, choose the reasoning where score is equal to the median score
        reasonings = {}
        if self.use_COT_anchor:
            for score_path in results:
                for doc_data in score_path.document_scores:
                    document_id = doc_data.document_id
                    relevance_score = doc_data.relevance_score
                    if document_id not in key_to_document:
                        continue
                    document_original_id = key_to_document[document_id]
                    if relevance_score == anchor_scores[document_original_id]:
                        reasonings[document_original_id] = doc_data.reasoning

        scored_anchors = [
            Anchor(
                docid=doc.docid,
                text=doc.text,
                score=anchor_scores.get(doc.docid, 0),
                reasoning=reasonings.get(doc.docid, ""),
            )
            for doc in anchors
        ]
        return scored_anchors

    @traceback_wrapper
    async def get_score_batch(
        self,
        query: str,
        batch: list[SearchResult],
        anchors: list[Anchor],
    ) -> Dict[str, List[float]]:
        # Get the prompt function
        get_score_prompt = ChunkwiseRankRelevancePrompt(
            query=query,
            anchors=anchors,
            documents=batch,
            use_COT=self.use_COT_document,
            num_anchor=self.num_anchor,
            output_schema=(
                DocumentRelevanceScoresBatchResponse
                if not self.use_COT_document
                else DocumentRelevanceScoresWithReasonBatchResponse
            ),
        )

        score_batch_response = await self.ask_llm(get_score_prompt)

        # make sure this works even if num_vote is 1, which causes the LLM to generate only one response, not a list
        if not isinstance(score_batch_response, MultiGenerationsResponse):
            score_batch_response = MultiGenerationsResponse(results=[score_batch_response])
        self.total_completion_tokens += score_batch_response.completion_tokens
        self.total_prompt_tokens += score_batch_response.prompt_tokens

        # gather the results from all generations
        gathered_scores = defaultdict(list)
        key_to_document = get_score_prompt.key_to_document
        results: list = score_batch_response.results
        for score_path in results:
            for doc_data in score_path.document_scores:
                document_id = doc_data.document_id
                relevance_score = doc_data.relevance_score
                if document_id not in key_to_document:
                    continue
                document_original_id = key_to_document[document_id]
                gathered_scores[document_original_id].append(relevance_score)

        return gathered_scores

    @abstractmethod
    async def ask_llm(self, get_score_prompt):
        pass

    @abstractmethod
    def truncate(self, text, length):
        pass


class OpenaiChunkRanker(BaseChunkwiseRanker):
    def __init__(
        self,
        model_name_or_path: str = "gpt-4o-mini",
        batch_size: int = 10,
        num_vote: int = 5,
        method: str = "random",
        temperature: float = 0.5,
        num_anchor: int = 5,
        use_COT: bool = True,
        use_COT_anchor: bool = True,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            num_vote=num_vote,
            method=method,
            temperature=temperature,
            num_anchor=num_anchor,
            use_COT=use_COT,
            use_COT_anchor=use_COT_anchor,
        )

        self.llm_client = LLMClient()
        self.tokenizer = tiktoken.encoding_for_model(model_name_or_path)
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def ask_llm(self, get_score_prompt):
        # # if not self.print_example:
        # #     print(f"*** System Prompt:\n{get_score_prompt.system()}")
        # #     print(f"*** User Prompt:\n{get_score_prompt.user()}")

        score_batch_response = await self.llm_client.astructure_completion_with_prompt(
            prompt_func=get_score_prompt,
            response_model=get_score_prompt.output_schema,
            model=self.llm,
            temperature=self.temperature,
            n=self.num_vote,  # self-consistency
            max_retries=3,
        )
        # if not self.print_example:
        #     self.print_example = True
        #     # print(f"*** System Prompt:\n{get_score_prompt.system()}")
        #     # print(f"*** User Prompt:\n{get_score_prompt.user()}")
        #     print(f"*** Output: {score_batch_response}")
        return score_batch_response

        completion = await self.client.beta.chat.completions.parse(
            model=self.llm,
            messages=[
                {"role": "system", "content": get_score_prompt.system()},
                {"role": "user", "content": get_score_prompt.user()},
            ],
            temperature=self.temperature,
            n=self.num_vote,
            response_format=get_score_prompt.output_schema,
        )

        results = []
        for choice in completion.choices:
            result = choice.message.parsed
            results.append(result)

        return MultiGenerationsResponse(
            results=results,
            completion_tokens=completion.usage.completion_tokens,
            prompt_tokens=completion.usage.prompt_tokens,
        )

    def truncate(self, text, length):
        return self.tokenizer.decode(self.tokenizer.encode(text)[:length])


class VLLMChunkRanker(BaseChunkwiseRanker):
    def __init__(
        self,
        model_name_or_path: str,
        base_url: str,
        batch_size: int = 10,
        num_vote: int = 5,
        method: str = "random",
        temperature: float = 0.5,
        num_anchor: int = 5,
        use_COT: bool = True,
        use_COT_anchor: bool = True,
        guided_decoding_backend: str = "outlines",
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            num_vote=num_vote,
            method=method,
            temperature=temperature,
            num_anchor=num_anchor,
            use_COT=use_COT,
            use_COT_anchor=use_COT_anchor,
        )

        self.llm_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""), base_url=base_url)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.guided_decoding_backend = guided_decoding_backend

    async def ask_llm(self, get_score_prompt):

        response_format_schema = get_score_prompt.output_schema
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=get_score_prompt.system()),
            ChatMessage(role=MessageRole.USER, content=get_score_prompt.user()),
        ]

        completion = await self.llm_client.beta.chat.completions.parse(
            model=self.llm,
            # messages=[
            #     {"role": "system", "content": get_score_prompt.system()},
            #     {"role": "user", "content": get_score_prompt.user()},
            # ],
            messages=messages,
            temperature=self.temperature,
            n=self.num_vote,
            response_format=response_format_schema,
            extra_body={
                "guided_json": response_format_schema.model_json_schema(),
                # "guided_decoding_backend": "lm-format-enforcer" or "outlines",
                "guided_decoding_backend": self.guided_decoding_backend,
            },
        )

        results = []
        for choice in completion.choices:
            result = choice.message.parsed
            results.append(result)

        return_obj = MultiGenerationsResponse(
            results=results,
            completion_tokens=completion.usage.completion_tokens,
            prompt_tokens=completion.usage.prompt_tokens,
        )
        if not self.print_example:
            self.print_example = True
            print(f"*** System Prompt:\n{messages[0].content}")
            print(f"*** User Prompt:\n{messages[1].content}")
            print(f"*** Output: {return_obj}")
        return return_obj

    def truncate(self, text, length):
        return self.tokenizer.decode(self.tokenizer.encode(text)[:length])
