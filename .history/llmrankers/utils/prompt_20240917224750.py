from abc import ABC, abstractmethod

from llm_schema import (
    Criteria,
    MatchingResult,
)
from string_utils import StringUtils


class BasePromptFunction(ABC):
    @abstractmethod
    def system(self) -> str:
        pass

    @abstractmethod
    def user(self) -> str:
        pass

    def assistant(self) -> str:
        pass


class SetwiseRankRelevancePrompt(BasePromptFunction):
    def __init__(
        self,
        patient_information: str,
        trials: list[dict],
        use_COT: bool = False,
        num_batch: int = 4,
    ) -> None:
        self.patient_information = patient_information
        self.trials = trials
        self.use_COT = use_COT
        self.num_batch = num_batch
        self.key_to_trial = {}

    def system(self):
        prompt = (
            "You are a helpful assistant for clinical trial recruitment. You will be given a patient note and a list of clinical trials.\n"
            "Your task is to output a relevance score (R) between 0 and 100 for each clinical trial. This score should reflect both how well the trial matches the patient's condition and how it compares to other trials in the list. A score of 0 means the trial is the least relevant among all trials for this patient, 80 might mean that it could be the most relevant among the trials but not exactly relevant to the patient note , while 100 means it's the most relevant relative to other trials and exactly to the patient note.\n"
        )
        if self.use_COT:
            prompt += (
                "Use the following step-by-step approach:\n"
                "1. For each trial, analyze its title and summary in relation to the patient's information. Provide a brief explanation and an initial relevance score (0-100).\n"
                "2. Compare the trials, explaining why some are more relevant than others. "
                f"To reduce the number of cross-trial comparisons, you can use the top {self.num_batch} trials to compare with other trials.\n"
                "3. Adjust the scores based on your comparison and output the final relevance score for each trial.\n"
                "Example:\n"
                "Trial 1:\n"
                "Title: 'A Clinical Trial for Heart Disease'\n"
                "Summary: 'This trial is designed to evaluate the effectiveness of a new drug in treating heart disease.'\n"
                "Trial 2:\n"
                "Title: 'A Clinical Trial for Cancer'\n"
                "Summary: 'This trial is designed to evaluate the effectiveness of a new drug in treating cancer.'"
                "\n-----\n"
                "Please provide your analysis and relevance scores:\n"
                "Trial 1: Initial Score: 70\n"
                "Explanation: The trial is related to heart disease, which is a common condition for this patient.\n"
                "Trial 2: Initial Score: 70\n"
                "Explanation: The trial is related to cancer, which is a common condition for this patient.\n"
                "Comparison: Trial 1 is more relevant to the patient than Trial 2 because it is related to a common condition for this patient. Increase trial 1's score and decrease trial 2's score.\n"
                "Final Relevance Scores: Trial 1 (80), Trial 2 (60)\n"
            )
        return prompt

    def user(self):
        user_prompt = f"Patient Information:\n{self.patient_information}\n\n"
        user_prompt += "Clinical Trials:\n"
        for i, trial in enumerate(self.trials, 1):
            self.key_to_trial[i] = trial["_id"]
            user_prompt += (
                f"Trial {i}:\n"
                f"Title: {trial['_source']['metadata']['brief_title']}\n"
                f"Summary: {trial['_source']['metadata']['brief_summary']['textblock']}\n"
            )

        user_prompt += (
            "\n-----\n" "Please provide your analysis and relevance scores:\n"
        )
        return user_prompt
