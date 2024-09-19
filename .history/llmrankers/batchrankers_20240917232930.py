from collections import defaultdict
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
    TrialScoringBatchResponse,
    BatchScoringResponseType,
    MultiGenerationsResponse,
    Model,
)


class BatchRanker:
    def __init__(
        self,
        model_name_or_path: str = "gpt-4o-mini",
        num_batch: int = 5,
        num_vote: int = 5,
        method: str = "random",
        temperature: float = 0.5,
    ):
        self.llm_client = LLMClient()
        self.num_batch = num_batch
        self.num_vote = num_vote
        self.method = method
        self.tokenizer = tiktoken.encoding_for_model(model_name_or_path)
        self.llm = model_name_or_path
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.temperature = temperature

    def rerank(
        self,
        query: str,
        ranking: List[SearchResult],
    ) -> List[SearchResult]:
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
            top_docs = random.sample(ranking, k=self.num_batch)
        elif self.method == "top":
            raise NotImplementedError

        remaining_docs = [doc for doc in ranking if doc not in top_docs]
        batch_size = len(remaining_docs) // self.num_batch

        # Step 2: Perform self-consistency COT batching on all batches again, but with previous anchors added to each batch.
        batches = [
            top_docs + remaining_docs[i : i + batch_size]
            for i in range(0, len(remaining_docs), batch_size)
        ]
        all_scoring_process = []
        for _, batch in enumerate(batches):
            _scoring_process = self.get_score_batch(
                query=query,
                batch=batch,
                num_batch=self.num_batch,
                num_vote=self.num_vote,
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

        final_ranking = {
            SearchResult(
                docid=doc_id,
                score=score,
                text="",
            )
            for doc_id, score in aggregated_scores.items()
        }

        return final_ranking

    async def get_score_batch(
        self,
        query: str,
        batch: list,
        num_batch: int = 5,
        num_vote: int = 5,
        temperature: float = 0.5,
    ) -> Dict[str, float]:
        try:
            # run the LLM self-consistency COT batching
            get_score_prompt = SetwiseRankRelevancePrompt(
                query=query,
                documents=batch,
                use_COT=True,
                num_batch=num_batch,
            )

            score_batch_response: BatchScoringResponseType = (
                await self.llm_client.astructure_completion_with_prompt(
                    prompt_func=get_score_prompt,
                    response_model=TrialScoringBatchResponse,
                    model=self.llm,
                    temperature=temperature,
                    n=num_vote,  # self-consistency
                )
            )
            # make sure this works even if num_vote is 1, which causes the LLM to generate only one response, not a list
            if not isinstance(score_batch_response, MultiGenerationsResponse):
                score_batch_response = MultiGenerationsResponse(
                    results=[score_batch_response],
                )

            # gather the results from all generations
            gathered_scores = defaultdict(list)
            key_to_trial = get_score_prompt.key_to_trial
            for score_path in score_batch_response:
                for trial_score in score_path.trial_scores:
                    trial_id = key_to_trial[trial_score.trial_number]
                    gathered_scores[trial_id].append(trial_score.relevance_score)

            return gathered_scores
        except Exception as e:
            logger.error(f"Error in get_score_batch: {e}")
            return {}
    
    def truncate(self, text, length):
        return self.tokenizer.decode(self.tokenizer.encode(text)[:length])