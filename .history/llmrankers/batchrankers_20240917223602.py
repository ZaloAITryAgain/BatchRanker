from collections import defaultdict
from .rankers import SearchResult
from typing import Dict, List
import statistics
import asyncio

from utils.prompt import SetwiseRankRelevancePrompt
from utils.llm_client import LLMClient
from utils.logging import logger
from utils.llm_schema import (
    TrialScoringBatchResponse,
    BatchScoringResponseType,
    MultiGenerationsResponse,
    RankingResult,
    Model,
)


class BatchRanker:
    def __init__(self):
        self.llm_client = LLMClient()
    
    def rerank(
        self,
        query: str,
        ranking: List[SearchResult],
    ) -> List[SearchResult]:
        pass

    async def setwise_rank_relevance(
        self,
        patient_information: str,
        trials: list,
        num_batch: int = 4,
    ) -> Dict[str, float]:
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
        top_trials = sorted(trials, key=lambda trial: trial["_score"], reverse=True)[
            :num_batch
        ]
        remaining_trials = [trial for trial in trials if trial not in top_trials]
        batch_size = len(remaining_trials) // num_batch

        # Step 2: Perform self-consistency COT batching on all batches again, but with previous anchors added to each batch.
        batches = [
            top_trials + remaining_trials[i : i + batch_size]
            for i in range(0, len(remaining_trials), batch_size)
        ]
        all_scoring_process = []
        for _, batch in enumerate(batches):
            _scoring_process = self.get_score_batch(
                patient_information,
                batch,
                num_batch,
            )
            all_scoring_process.append(_scoring_process)

        batched_scores = await asyncio.gather(*[score for score in all_scoring_process])

        # Step 3: Aggregate the results from all batches and return the final ranking.
        aggregated_scores = defaultdict(list)
        for score_dict in batched_scores:
            for trial_id, score_list in score_dict.items():
                aggregated_scores[trial_id].extend(score_list)

        aggregated_scores = {
            trial_id: statistics.median(scores)
            for trial_id, scores in aggregated_scores.items()
        }

        final_ranking = {
            trial_id: RankingResult(
                relevance_score_R=score,
                eligibility_score_E=0.0,  # TODO: test with eligibility = relevance
                eligibility_explanation="",
                relevance_explanation="",
            )
            for trial_id, score in aggregated_scores.items()
        }

        return final_ranking

    async def get_score_batch(
        self,
        patient_information: str,
        batch: list,
        num_batch: int = 10,
        num_vote: int = 5,
    ) -> Dict[str, float]:
        try:
            # run the LLM self-consistency COT batching
            get_score_prompt = SetwiseRankRelevancePrompt(
                patient_information=patient_information,
                trials=batch,
                use_COT=True,
                num_batch=num_batch,
            )

            score_batch_response: BatchScoringResponseType = (
                await self.llm_client.astructure_completion_with_prompt(
                    prompt_func=get_score_prompt,
                    response_model=TrialScoringBatchResponse,
                    model=Model.openai_gpt_4o_mini,
                    temperature=0.5,
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