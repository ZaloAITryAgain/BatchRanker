from abc import ABC, abstractmethod
from ..rankers import SearchResult


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
        query: str,
        documents: list[SearchResult],
        use_COT: bool = False,
        num_anchor: int = 4,
    ) -> None:
        self.query = query
        self.documents = documents
        self.use_COT = use_COT
        self.num_anchor = num_anchor
        self.key_to_document = {}

    def system(self):
        prompt = (
            "You are a helpful assistant for document ranking. You will be given a query and a list of documents.\n"
            "Your task is to output a relevance score (R) between 0 and 100 for each document. "
            "This score should reflect both how well the document matches the query and how it compares to other documents in the list. "
            "A score of 0 means the document is the least relevant among all documents for this query, "
            "80 might mean that it could be the most relevant among the documents but not exactly relevant to the query, "
            "while 100 means it's the most relevant relative to other documents and exactly to the query.\n"
        )
        # prompt = (
        #     "You are an expert in document ranking. Given a query and a list of documents, your task is to:\n"
        #     "1. Assess each document's relevance to the query.\n"
        #     "2. Compare the relevance of documents to each other.\n"
        #     "3. Assign a relevance score (R) between 0 and 100 for each document.\n\n"
        #     "Scoring guidelines:\n"
        #     "- 0: Completely irrelevant to the query and least relevant among all documents.\n"
        #     "- 1-20: Barely relevant, mentions query terms but out of context.\n"
        #     "- 21-40: Somewhat relevant, touches on the query topic but lacks depth.\n"
        #     "- 41-60: Moderately relevant, addresses the query but may be missing key aspects.\n"
        #     "- 61-80: Highly relevant, covers the query well but may not be the best match.\n"
        #     "- 81-99: Extremely relevant, almost perfectly matches the query.\n"
        #     "- 100: Perfect match, most relevant to the query and best among all documents.\n\n"
        #     "Consider both absolute relevance to the query and relative relevance compared to other documents.\n"
        # )
        if self.use_COT:
            prompt += (
                "Use the following step-by-step approach:\n"
                "1. For each document, analyze its text in relation to the query. Provide a brief explanation and an initial relevance score (0-100).\n"
                "2. Compare the documents, explaining why some are more relevant than others. "
                f"To reduce the number of cross-document comparisons, you can use the top {self.num_anchor} documents to compare with other documents.\n"
                "3. Adjust the scores based on your comparison and output the final relevance score for each document.\n"
                "Example:\n"
                "Document 1:\n"
                "Text: 'This document covers the basics of machine learning algorithms and their applications.'\n"
                "Document 2:\n"
                "Text: 'This document provides an overview of data science concepts and methodologies.'"
                "\n-----\n"
                "Please provide your analysis and relevance scores:\n"
                "Document 1: Initial Score: 70\n"
                "Explanation: The document is related to machine learning, which is relevant to the query.\n"
                "Document 2: Initial Score: 70\n"
                "Explanation: The document is related to data science, which is relevant to the query.\n"
                "Comparison: Document 1 is more relevant to the query than Document 2 because it focuses specifically on machine learning. Increase Document 1's score and decrease Document 2's score.\n"
                "Final Relevance Scores: Document 1 (75), Document 2 (65)\n"
            )
        else:
            prompt += (
                "Output a list of relevance scores for each document.\n"
            )
        return prompt

    def user(self):
        user_prompt = f"Query:\n{self.query}\n\n"
        user_prompt += "Documents:\n"
        for i, document in enumerate(self.documents, 1):
            self.key_to_document[i] = document.docid
            user_prompt += (
                f"Document {i}:\n"
                f"Text: {document.text}\n"
            )

        if self.use_COT:
            user_prompt += (
                "\n-----\n" "Please provide your analysis and relevance scores:\n"
            )
        else:
            user_prompt += (
                "\n-----\n" "Please provide your relevance scores:\n"
            )
        return user_prompt
