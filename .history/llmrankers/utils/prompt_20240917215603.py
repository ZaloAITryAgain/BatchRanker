from abc import ABC, abstractmethod

from service_platform.service.trial_gpt.schema import (
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


class KeywordGenerationPrompt(BasePromptFunction):
    def __init__(
        self,
        patient_information: str,
    ) -> None:
        self.patient_information = patient_information

    def system(self) -> str:
        system = "You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description. "
        system += "Please first summarize the main medical problems of the patient. "
        system += "Then generate up to 32 key conditions for searching relevant clinical trials for this patient. The key condition list should be ranked by priority. "
        system += 'Please output only a JSON dict formatted as Dict{{"summary": Str(summary), "conditions": List[Str(condition)]}}.'
        return system

    def user(self) -> str:
        user = (
            f"Here is the my description: \n{self.patient_information}\n\nJSON output:"
        )
        return user


class SummaryGenerationPrompt(BasePromptFunction):
    def __init__(
        self,
        patient_information: str,
    ) -> None:
        self.patient_information = patient_information

    def system(self) -> str:
        system = "You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description. "
        system += "Please generate a comprehensive and descriptive summary for the main medical problems of the patient. "
        system += "Wrap the disease name between two *. "
        system += "For example: "
        system += "Patient is experiencing a severe *headache* and general *malaise*. "
        system += "The patient is diagnosed with *cancer*. "
        system += "The patient has a pain in *stomach*. "
        system += "Please output only a string result."
        return system

    def user(self) -> str:
        user = f"Here is the my description: \n{self.patient_information}\n\nOutput:"
        return user


class MatchingPrompt(BasePromptFunction):
    def __init__(
        self,
        trial_info: dict,
        inc_exc: Criteria,
        patient_information: str,
    ) -> None:
        self.trial_info = trial_info
        self.patient_information = patient_information
        self.inc_exc = inc_exc.value

    def print_trial(self) -> str:
        """Given a dict of trial information, returns a string of trial."""

        trial = f"Title: {self.trial_info['brief_title']}\n"
        # trial += f"Target diseases: {', '.join(trial_info['diseases_list'])}\n"
        # trial += f"Interventions: {', '.join(trial_info['drugs_list'])}\n"
        trial += f"Summary: {self.trial_info['brief_summary']['textblock']}\n"

        if self.inc_exc == "inclusion":
            trial += (
                "Inclusion criteria:\n %s\n" % self.trial_info["criteria"]["inclusion"]
            )
        elif self.inc_exc == "exclusion":
            trial += (
                "Exclusion criteria:\n %s\n" % self.trial_info["criteria"]["exclusion"]
            )

        return trial

    def system(
        self,
    ):
        prompt = f"You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient note and the {self.inc_exc} criteria of a clinical trial to determine the patient's eligibility at the criterion level.\n"

        if self.inc_exc == "inclusion":
            prompt += "The factors that allow someone to participate in a clinical study are called inclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.\n"

        elif self.inc_exc == "exclusion":
            prompt += "The factors that disqualify someone from participating are called exclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.\n"

        prompt += f"You should check the {self.inc_exc} criteria one-by-one, and output the following three elements for each criterion:\n"
        prompt += f"\tElement 1. For each {self.inc_exc} criterion, briefly generate your reasoning process: First, judge whether the criterion is not applicable (not very common), where the patient does not meet the premise of the criterion. Then, check if the patient note contains direct evidence. If so, judge whether the patient meets or does not meet the criterion. If there is no direct evidence, try to infer from existing evidence, and answer one question: If the criterion is true, is it possible that a good patient note will miss such information? If impossible, then you can assume that the criterion is not true. Otherwise, there is not enough information.\n"
        prompt += "\tElement 2. If there is relevant information, you must generate a list of relevant sentence IDs in the patient note. If there is no relevant information, you must annotate an empty list.\n"
        prompt += f"\tElement 3. Classify the patient eligibility for this specific {self.inc_exc} criterion: "

        if self.inc_exc == "inclusion":
            prompt += 'the label must be chosen from {"not applicable", "not enough information", "included", "not included"}. "not applicable" should only be used for criteria that are not applicable to the patient. "not enough information" should be used where the patient note does not contain sufficient information for making the classification. Try to use as less "not enough information" as possible because if the note does not mention a medically important fact, you can assume that the fact is not true for the patient. "included" denotes that the patient meets the inclusion criterion, while "not included" means the reverse.\n'
        elif self.inc_exc == "exclusion":
            prompt += 'the label must be chosen from {"not applicable", "not enough information", "excluded", "not excluded"}. "not applicable" should only be used for criteria that are not applicable to the patient. "not enough information" should be used where the patient note does not contain sufficient information for making the classification. Try to use as less "not enough information" as possible because if the note does not mention a medically important fact, you can assume that the fact is not true for the patient. "excluded" denotes that the patient meets the exclusion criterion and should be excluded in the trial, while "not excluded" means the reverse.\n'

        prompt += "You should output only a JSON dict exactly formatted as: dict{str(criterion_number): list[str(element_1_brief_reasoning), list[int(element_2_sentence_id)], str(element_3_eligibility_label)]}."

        return prompt

    def user(self):
        user_prompt = f"Here is the my note, each sentence is led by a sentence_id:\n{self.patient_information}\n\n"
        user_prompt += f"Here is the clinical trial:\n{self.print_trial()}\n\n"
        user_prompt += "Plain JSON output:"

        return user_prompt


class BatchMatchingPrompt(BasePromptFunction):
    def __init__(
        self,
        trials: list[dict],
        criterion_type: Criteria,
        patient_information: str,
    ) -> None:
        self.trials = trials
        self.patient_information = patient_information
        self.criterion_type = criterion_type
        self.key_to_trial = {}

    def print_criteria(self) -> str:
        """Given a list of criteria, returns a string of criteria."""

        criteria = ""
        index = 1
        for trial in self.trials:
            # parse the criteria string to list
            criteria_list = trial["_source"]["metadata"]["criteria"][
                self.criterion_type
            ].split("\n")

            # sanitize the criteria list
            criteria_list = list(map(StringUtils.sanitize_criterion, criteria_list))

            # iterate through the criteria list and update criteria string + the key_to_trial dict
            for criterion in criteria_list:
                # update the key_to_trial dict
                self.key_to_trial[index] = trial["_id"]

                # update the criteria string
                criteria += f"Criterion {index}: {criterion}\n"

                # update the index
                index += 1

        return criteria

    def system(self):
        prompt = f"You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient note and a list of {self.criterion_type.value} criteria of clinical trials to determine the patient's eligibility at the criterion level.\n"

        if self.criterion_type == Criteria.INCLUSION:
            prompt += "The factors that allow someone to participate in a clinical study are called inclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.\n"

        elif self.criterion_type == Criteria.EXCLUSION:
            prompt += "The factors that disqualify someone from participating are called exclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.\n"

        prompt += f"You should check the {self.criterion_type.value} criteria one-by-one, and classify the patient eligibility for each criterion: "

        if self.criterion_type == Criteria.INCLUSION:
            prompt += (
                'the label must be chosen from {"Y", "N", "NA", "NE"}. \n'
                '- "Y" means the patient meets the inclusion criterion. \n'
                '- "N" means the criterion is relevant to the patient, but the patient does not meet it based on the information provided. \n'
                '- "NA" should only be used for criteria that are completely irrelevant or impossible to apply to the patient (e.g., pregnancy-related criteria for male patients).\n'
                '- "NE" means the criterion is not evaluable due to being unclear, lacking context, or containing gibberish data.\n'
                '**Notes:**\n'
                '- If the patient note does not contain sufficient information for making the classification, lean towards "N" unless the criterion is clearly not applicable or not evaluable.\n'
                '- You don\'t need to output "NE" if the criterion is not evaluable, simply skip it.'
            )

        elif self.criterion_type == Criteria.EXCLUSION:
            prompt += 'the label must be chosen from {"not applicable", "not enough information", "excluded", "not excluded"}. "not applicable" should only be used for criteria that are not applicable to the patient. "not enough information" should be used where the patient note does not contain sufficient information for making the classification. Try to use as less "not enough information" as possible because if the note does not mention a medically important fact, you can assume that the fact is not true for the patient. "excluded" denotes that the patient meets the exclusion criterion and should be excluded in the trial, while "not excluded" means the reverse.\n'

        return prompt

    def user(self):
        user_prompt = f"Here is the patient note:\n{self.patient_information}\n\n"
        user_prompt += f"Here is the list of {self.criterion_type.value} criteria:\n{self.print_criteria()}\n\n"
        user_prompt += "Output:"

        return user_prompt


class RankingPrompt(BasePromptFunction):
    def __init__(
        self,
        patient_information: str,
        matching_results: MatchingResult,
        trial_info: dict,
    ) -> None:
        self.matching_results = matching_results
        self.trial_info = trial_info
        self.patient_information = patient_information

    @staticmethod
    def convert_criteria_pred_to_string(
        matching_results: MatchingResult,
        trial_info: dict,
    ) -> str:
        """Given the TrialGPT matching results, output the linear string of the criteria."""
        output = ""

        for inc_exc in Criteria.to_list():
            # first get the idx2criterion dict
            idx2criterion = {}
            criteria = trial_info["criteria"][inc_exc.value].split("\n\n")

            idx = 0
            for criterion in criteria:
                criterion = criterion.strip()

                if (
                    "inclusion criteria" in criterion.lower()
                    or "exclusion criteria" in criterion.lower()
                ):
                    continue

                if len(criterion) < 5:
                    continue

                idx2criterion[str(idx)] = criterion
                idx += 1

            for idx, info in enumerate(
                getattr(matching_results, inc_exc.value).items(),
            ):
                criterion_idx, preds = info

                if criterion_idx not in idx2criterion:
                    continue

                criterion = idx2criterion[criterion_idx]

                if len(preds) != 3:
                    continue

                output += f"{inc_exc.value} criterion {idx}: {criterion}\n"
                output += f"\tPatient relevance: {preds[0]}\n"
                if len(preds[1]) > 0:
                    output += f"\tEvident sentences: {preds[1]}\n"
                output += f"\tPatient eligibility: {preds[2]}\n"

        return output

    def system(self):
        prompt = "You are a helpful assistant for clinical trial recruitment. You will be given a patient note, a clinical trial, and the patient eligibility predictions for each criterion.\n"
        prompt += "Your task is to output two scores, a relevance score (R) and an eligibility score (E), between the patient and the clinical trial.\n"
        prompt += "First explain the consideration for determining patient-trial relevance. Predict the relevance score R (0~100), which represents the overall relevance between the patient and the clinical trial. R=0 denotes the patient is totally irrelevant to the clinical trial, and R=100 denotes the patient is exactly relevant to the clinical trial.\n"
        prompt += "Then explain the consideration for determining patient-trial eligibility. Predict the eligibility score E (-R~R), which represents the patient's eligibility to the clinical trial. Note that -R <= E <= R (the absolute value of eligibility cannot be higher than the relevance), where E=-R denotes that the patient is ineligible (not included by any inclusion criteria, or excluded by all exclusion criteria), E=R denotes that the patient is eligible (included by all inclusion criteria, and not excluded by any exclusion criteria), E=0 denotes the patient is neutral (i.e., no relevant information for all inclusion and exclusion criteria).\n"
        prompt += 'Please output a JSON dict formatted as Dict{"relevance_explanation": Str, "relevance_score_R": Float, "eligibility_explanation": Str, "eligibility_score_E": Float}.'
        return prompt

    def user(self):
        # get the trial string
        trial = f"Title: {self.trial_info['brief_title']}\n"
        # trial += f"Target diseases: {', '.join(trial_info['diseases_list'])}\n"
        trial += f"Summary: {self.trial_info['brief_summary']['textblock']}\n"

        # then get the prediction strings
        pred = self.convert_criteria_pred_to_string(
            self.matching_results,
            self.trial_info,
        )

        user_prompt = "Here is the my note:\n"
        user_prompt += self.patient_information + "\n\n"
        user_prompt += "Here is the clinical trial description:\n"
        user_prompt += trial + "\n\n"
        user_prompt += "Here are the criterion-levle eligibility prediction:\n"
        user_prompt += pred + "\n\n"
        user_prompt += "Plain JSON output:"

        return user_prompt


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


class GenerateQuestionPrompt(BasePromptFunction):
    def __init__(
        self,
        patient_information: str,
        trials: list[dict],
        previous_questions: list[str],
    ) -> None:
        self.patient_information = patient_information
        self.trials = trials
        self.previous_questions = previous_questions

    def system(self):
        # construct the prompt
        prompt = "You are a helpful assistant for clinical trial recruitment. You will be given a list of inclusion and exclusion criteria of each trials"
        if self.previous_questions:
            prompt += ", the user information, and previous questions.\n"
        else:
            prompt += ", and the user information.\n"
        prompt += "Your task is to create 3 questions that help to enrich user information based on inclusion and exclusion.\n"
        prompt += "First, consider each inclusion and exclusion step by step to identify five questions that help separate the trials as clearly as possible.\n"
        prompt += "For information already provided in the user information, you do not need to create questions related to these provided sections.\n"
        if self.previous_questions:
            prompt += "You are not allow to create duplicate questions of previous questions, even in terms of meaning.\n"
        prompt += "Keep your questions as simple and general as possible.\n"
        prompt += "Please output a List formatted as ['question_1', 'question_2', 'question_3']\n"
        prompt += (
            "You do not need to provide any additional information or characters.\n"
        )
        return prompt

    def user(self):
        user_prompt = "Here is list of inclusion and exclusion criteria:\n"
        for i in range(len(self.trials)):
            user_prompt += (
                f"Trial {i+1}. {self.trials[i]['_source']['metadata']['criteria']}\n"
            )
        user_prompt += (
            f"Here is the patient description: \n{self.patient_information}\n"
        )
        if self.previous_questions:
            user_prompt += f"Here is previous questions: \n{self.previous_questions}\n"
        user_prompt += "List output:"

        return user_prompt


class GeneratePatientInformationPrompt(BasePromptFunction):
    def __init__(
        self,
        patient_information: str,
        message_history: list[dict],
    ) -> None:
        self.patient_information = patient_information
        self.message_history = message_history

    def system(self):
        # construct the prompt
        prompt = "You are a helpful assistant for clinical trial recruitment. You will be given a history of conversation between doctor and patient along with previous patient descriptions\n"
        prompt += "Your task is to synthesize new information from the conversation history together with the previous patient description to create a new, more complete description.\n"
        prompt += (
            "You do not need to provide any additional information or characters.\n"
        )
        return prompt

    def user(self):
        user_prompt = "Here is a history of conversation:\n"
        for message in self.message_history:
            user_prompt += f"Question: {message['question']}"
            user_prompt += f"Answer: {message['answer']}"

        user_prompt += f"Here is the my description: \n{self.patient_information}\n"
        user_prompt += "New patient information:"
        return user_prompt


class SummaryTrialPrompt(BasePromptFunction):
    def __init__(
        self,
        clinic_trial: str,
    ) -> None:
        self.clinic_trial = clinic_trial

    def system(self) -> str:
        system = "You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description. "
        system += "Please summarize this clinical trial, your summarization should not exceed 350 words."
        system += 'Please output only a JSON dict formatted as Dict{{"summary": Str(summary)}}.'
        return system

    def user(self) -> str:
        user = f"Here is the my clinical trial: \n{self.clinic_trial}\n\nJSON output:"
        return user


class SummaryTrialWithCriteriaPrompt(BasePromptFunction):
    def __init__(
        self,
        clinic_trial: str,
        list_criteria: list[str],
    ) -> None:
        self.clinic_trial = clinic_trial
        self.list_criteria = list_criteria

    def system(self) -> str:
        system = "You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description. "
        system += "Please first  summarize this clinical trial, your summarization should not exceed 350 words."
        system += "Then, generate keywords in this trial, corresponding to the criteria I provided."
        system += 'Please output only a JSON dict formatted as Dict{{"summary": Str(summary), Str(criteria): Str(keyword)}}.'
        return system

    def user(self) -> str:
        user = (
            f"Here is the list of criteria:\n{self.list_criteria}\n\n"
            + f"Here is the my clinical trial: \n{self.clinic_trial}\n\nJSON output:"
        )
        return user


class ExtractCriteriaPrompt(BasePromptFunction):
    def __init__(
        self,
        list_criteria: str,
    ) -> None:
        self.list_criteria = list_criteria

    def system(self) -> str:
        system = (
            "You are a helpful assistant for clinical trial search and your task is find some keywords of criteria in that text to help searching with rules based.\n"
            + "There is a list of inclusion/exclusion criteria, I want you to extract common criteria in that list. "
            + "Criteria should be keywords such as age, gender, married, diabetes, cancer, BMI, etc. "
            + 'Please output only a JSON dict formatted as Dict{{"common_inclusion_criteria": List[Str(criteria)], "common_exclusion_criteria": List[Str(criteria)]}}.'
        )
        return system

    def user(self) -> str:
        user = f"Here is the list criteria of trials: \n{self.list_criteria}\n\nJSON output:"
        return user


class ExtractKeywordPrompt(BasePromptFunction):
    def __init__(
        self,
        clinic_trial: str,
        list_criteria: list[str],
    ) -> None:
        self.clinic_trial = clinic_trial
        self.list_criteria = list_criteria

    def system(self) -> str:
        system = "You are a helpful assistant and your task is to help find the keywords in this trial, corresponding to the criteria I provided. "
        system += "Please output only a JSON dict formatted as Dict{{Str(criteria): Str(keyword)]}}."
        return system

    def user(self) -> str:
        user = (
            f"Here is the list of criteria:\n{self.list_criteria}\n\n"
            + f"Here is the my clinical trial: \n{self.clinic_trial}\n\nJSON output:"
        )
        return user


# user profile processor


class ProfileSummarizationPrompt(BasePromptFunction):
    def __init__(
        self,
        user_input_profile: str,
        predefined_criteria: list[str] = None,
    ) -> None:
        self.user_input_profile = user_input_profile
        self.predefined_criteria = predefined_criteria

    def system(self) -> str:
        system = "You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description. "
        system += "Please summarize the main medical problems of the patient. "
        system += 'Please output only a JSON dict formatted as Dict{{"summary": Str(summary)}}.'
        return system

    def user(self) -> str:
        user = (
            f"Here is the list of criteria:\n{self.predefined_criteria}\n\n"
            + f"Here is the my description: \n{self.user_input_profile}\n\nJSON output:"
        )
        return user


class KeywordExtractionPrompt(BasePromptFunction):
    def __init__(
        self,
        user_input_profile: str,
        predefined_criteria: list[str] = None,
    ) -> None:
        self.user_input_profile = user_input_profile
        self.predefined_criteria = predefined_criteria

    def system(self) -> str:
        system = "You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description. "
        system += "Please generate criteria for searching relevant clinical trials for this patient."
        system += 'Please output only a JSON dict formatted as Dict{{ "criteria": Dict{{ Str(criteria): Str(keyword)}} }}.'
        return system

    def user(self) -> str:
        user = (
            f"Here is the list of criteria:\n{self.predefined_criteria}\n\n"
            + f"Here is the my description: \n{self.user_input_profile}\n\nJSON output:"
        )
        return user


class ProfileSummarizationAndKeywordExtractionPrompt(BasePromptFunction):
    def __init__(
        self,
        user_input_profile: str,
        predefined_criteria: list[str] = None,
    ) -> None:
        self.user_input_profile = user_input_profile
        self.predefined_criteria = predefined_criteria

    def system(self) -> str:
        system = "You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description. "
        system += "Please first summarize the main medical problems of the patient. "
        system += "Then generate criteria for searching relevant clinical trials for this patient."
        system += 'Please output only a JSON dict formatted as Dict{{"summary": Str(summary), Str(criteria): Str(keyword)}}.'
        return system

    def user(self) -> str:
        user = (
            f"Here is the list of criteria:\n{self.predefined_criteria}\n\n"
            + f"Here is the my description: \n{self.user_input_profile}\n\nJSON output:"
        )
        return user


class GetTopTrialPrompt(BasePromptFunction):
    def __init__(
        self,
        patient_information: str,
        list_trials: list,
        num_top_trial: int = 5,
    ) -> None:
        self.list_trials = list_trials
        self.patient_information = patient_information
        self.num_top_trial = num_top_trial

    def print_trial(self, trial_index) -> str:
        trial_info = self.list_trials[trial_index]
        trial = f"trial_id: {trial_index}\n"
        trial += f"Title: {trial_info['brief_title']}\n"
        trial += f"Summary: {trial_info['brief_summary']['textblock']}\n"
        trial += "Inclusion criteria:\n %s\n" % trial_info["criteria"]["inclusion"]
        trial += "Exclusion criteria:\n %s\n" % trial_info["criteria"]["exclusion"]
        return trial

    def system(
        self,
    ):
        prompt = "You are a helpful assistant for clinical trial search.\n"
        prompt += "I give you a patient note and list of clinical trials. "
        prompt += f"You need to find me the top {self.num_top_trial} eligible clinical trials for this patient, with the better trial coming first."
        prompt += "You must consider carefully on in Inclusion and Exclusion criteria of each trial."
        # prompt += 'You should output only a JSON dict exactly formatted as: Dict{{"top_5_trial_id": List(trial_id), "top_5_trial_title": List(trial_title)}}.'
        prompt += 'You should output only a JSON dict exactly formatted as: Dict{{"top_trial_id": List[Int(trial_id)]}}.'

        return prompt

    def user(self):
        input_list_trials = "\n".join(
            [self.print_trial(i) for i in range(len(self.list_trials))],
        )
        user_prompt = f"Here is the patient note:\n{self.patient_information}\n\n"
        user_prompt += f"Here is list clinical trial:\n{input_list_trials}\n\n"
        user_prompt += "Plain JSON output:"

        return user_prompt
