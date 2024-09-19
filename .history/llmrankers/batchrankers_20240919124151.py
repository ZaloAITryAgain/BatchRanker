from collections import defaultdict
import time
from .rankers import SearchResult
from typing import Dict, List
import statistics
import tiktoken
import asyncio
import random

from llmrankers.utils.prompt import SetwiseRankRelevancePrompt
from llmrankers.utils.llm_client import LLMClient
from llmrankers.utils.logging import logger
from llmrankers.utils.llm_schema import (
    TrialScoringNoCOTBatchResponse,
    TrialScoringBatchResponse,
    BatchScoringResponseType,
    MultiGenerationsResponse,
    Model,
)


class BatchRanker:
    def __init__(
        self,
        model_name_or_path: str = "gpt-4o-mini",
        batch_size: int = 10,
        num_vote: int = 5,
        method: str = "random",
        temperature: float = 0.5,
        num_anchor: int = 5,
        use_COT: bool = True,
    ):
        self.llm_client = LLMClient()
        self.batch_size = batch_size
        self.num_vote = num_vote
        self.method = method
        self.tokenizer = tiktoken.encoding_for_model(model_name_or_path)
        self.llm = model_name_or_path
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.temperature = temperature
        self.num_anchor = num_anchor
        self.use_COT = use_COT

    def rerank(
        self,
        query: str,
        ranking: List[SearchResult],
    ) -> List[SearchResult]:
        # time.sleep(2)
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        return asyncio.run(self._rerank(query, ranking))

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
            top_docs = sorted(ranking, key=lambda x: x.score, reverse=True)[:self.num_anchor]
        elif self.method == "none":
            top_docs = []
        else:
            raise NotImplementedError

        remaining_docs = [doc for doc in ranking if doc not in top_docs]

        # Step 2: Perform self-consistency COT batching on all batches again, but with previous anchors added to each batch.
        batches = [
            top_docs + remaining_docs[i : i + self.batch_size]
            for i in range(0, len(remaining_docs), self.batch_size)
        ]
        all_scoring_process = []
        for _, batch in enumerate(batches):
            _scoring_process = self.get_score_batch(
                query=query,
                batch=batch,
            )
            all_scoring_process.append(_scoring_process)

        batched_scores = await asyncio.gather(*[score for score in all_scoring_process])

        # Step 3: Aggregate the results from all batches and return the final ranking.
        aggregated_scores = defaultdict(list)
        for score_dict in batched_scores:
            for doc_id, score_list in score_dict.items():
                aggregated_scores[doc_id].extend(score_list)

        aggregated_scores = {
            doc_id: statistics.median(scores)
            for doc_id, scores in aggregated_scores.items()
        }

        final_ranking = [
            SearchResult(
                docid=doc_id,
                score=score,
                text="",
            )
            for doc_id, score in aggregated_scores.items()
        ]

        return sorted(final_ranking, key=lambda x: x.score, reverse=True)

    async def get_score_batch(
        self,
        query: str,
        batch: List[SearchResult],
    ) -> Dict[str, List[float]]:
        try:
            # run the LLM self-consistency COT batching
            get_score_prompt = SetwiseRankRelevancePrompt(
                query=query,
                documents=batch,
                use_COT=self.use_COT,
                num_anchor=self.num_anchor,
            )

            score_batch_response: BatchScoringResponseType = (
                await self.llm_client.astructure_completion_with_prompt(
                    prompt_func=get_score_prompt,
                    response_model=TrialScoringBatchResponse if self.use_COT else TrialScoringNoCOTBatchResponse,
                    model=self.llm,
                    temperature=self.temperature,
                    n=self.num_vote,  # self-consistency
                    max_retries=3,
                )
            )
            # print(f"score_batch_response: {score_batch_response}")
            # make sure this works even if num_vote is 1, which causes the LLM to generate only one response, not a list
            if not isinstance(score_batch_response, MultiGenerationsResponse):
                score_batch_response = MultiGenerationsResponse(
                    results=[score_batch_response],
                )
            print(f"Total completion tokens: {score_batch_response.completion_tokens}")
            print(f"Total prompt tokens: {score_batch_response.prompt_tokens}")
            self.total_completion_tokens += score_batch_response.completion_tokens
            self.total_prompt_tokens += score_batch_response.prompt_tokens

            # gather the results from all generations
            gathered_scores = defaultdict(list)
            key_to_document = get_score_prompt.key_to_document
            for score_path in score_batch_response:
                for trial_score in score_path.trial_scores:
                    trial_number, relevance_score = trial_score.trial_number, trial_score.relevance_score
                    if trial_number not in key_to_document:
                        continue
                    trial_id = key_to_document[trial_number]
                    gathered_scores[trial_id].append(relevance_score)

            return gathered_scores
        except Exception as e:
            logger.error(f"Error in get_score_batch: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def truncate(self, text, length):
        return self.tokenizer.decode(self.tokenizer.encode(text)[:length])