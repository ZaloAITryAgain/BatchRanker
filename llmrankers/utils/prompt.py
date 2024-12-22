from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Type, Union
from pydantic import BaseModel
from pydantic.fields import Field

# from ..rankers import SearchResult
# from llmrankers.utils.llm_schema import DocumentRelevanceScoresBatchResponse


@dataclass
class SearchResult:
    docid: str
    score: float
    text: str


class BasePromptFunction(ABC):
    @abstractmethod
    def system(self) -> str:
        pass

    @abstractmethod
    def user(self) -> str:
        pass

    def assistant(self) -> str:
        return ""


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
            prompt += "Output a list of relevance scores for each document.\n"
        return prompt

    def user(self):
        user_prompt = f"Query:\n{self.query}\n\n"
        user_prompt += "Documents:\n"
        for i, document in enumerate(self.documents, 1):
            self.key_to_document[i] = document.docid
            user_prompt += f"Document {i}:\n" f"Text: {document.text}\n"

        if self.use_COT:
            user_prompt += "\n-----\n" "Please provide your analysis and relevance scores:\n"
        else:
            user_prompt += "\n-----\n" "Please provide your relevance scores:\n"
        return user_prompt


# class BatchwiseRankRelevancePrompt(BasePromptFunction):
#     def __init__(
#         self,
#         query: str,
#         documents: list[SearchResult],
#         use_COT: bool = False,
#         num_anchor: int = 4,
#         output_schema: Optional[BaseModel] = None,
#         **kwargs,
#     ):
#         super(BatchwiseRankRelevancePrompt, self).__init__(**kwargs)
#         self.query = query
#         self.documents = documents
#         self.use_COT = use_COT
#         self.num_anchor = num_anchor
#         self.output_schema = output_schema

#         self.key_to_document = {}

#     def system(self):
#         prompt = (
#             "You are a helpful assistant for document ranking. You will be given a query and a list of documents.\n"
#             "Your task is to output a relevance score (R) between 0 and 100 for each document. "
#             "This score should reflect both how well the document matches the query and how it compares to other documents in the list. "
#             "A score of 0 means the document is the least relevant among all documents for this query, "
#             "80 might mean that it could be the most relevant among the documents but not exactly relevant to the query, "
#             "while 100 means it's the most relevant relative to other documents and exactly to the query.\n\n"
#             "Scoring guidelines:\n"
#             "- 0: Completely irrelevant to the query and least relevant among all documents.\n"
#             "- 1-20: Barely relevant, mentions query terms but out of context.\n"
#             "- 21-40: Somewhat relevant, touches on the query topic but lacks depth.\n"
#             "- 41-60: Moderately relevant, addresses the query but may be missing key aspects.\n"
#             "- 61-80: Highly relevant, covers the query well but may not be the best match.\n"
#             "- 81-99: Extremely relevant, almost perfectly matches the query.\n"
#             "- 100: Perfect match, most relevant to the query and best among all documents.\n\n"
#             "Consider both absolute relevance to the query and relative relevance compared to other documents.\n"
#         )
#         if self.use_COT:
#             prompt += (
#                 "Use the following step-by-step approach:\n"
#                 "1. For each document, analyze its text in relation to the query. Provide a brief explanation and an initial relevance score (0-100).\n"
#                 "2. Compare the documents, explaining why some are more relevant than others. "
#                 f"To reduce the number of cross-document comparisons, you can use the top {self.num_anchor} documents to compare with other documents.\n"
#                 "3. Adjust the scores based on your comparison and output the final relevance score for each document.\n"
#                 "Example:\n"
#                 "Document 1:\n"
#                 "Text: 'This document covers the basics of machine learning algorithms and their applications.'\n"
#                 "Document 2:\n"
#                 "Text: 'This document provides an overview of data science concepts and methodologies.'"
#                 "\n-----\n"
#                 "Please provide your analysis and relevance scores:\n"
#                 "Document 1: Initial Score: 70\n"
#                 "Explanation: The document is related to machine learning, which is relevant to the query.\n"
#                 "Document 2: Initial Score: 70\n"
#                 "Explanation: The document is related to data science, which is relevant to the query.\n"
#                 "Comparison: Document 1 is more relevant to the query than Document 2 because it focuses specifically on machine learning. Increase Document 1's score and decrease Document 2's score.\n"
#                 "Final Relevance Scores: Document 1 (75), Document 2 (65)\n"
#             )
#         else:
#             prompt += f"To output a list of relevance scores for each document, you need to followthe following schema: {self.output_schema.model_json_schema()}\n"
#         return prompt

#     def user(self):
#         user_prompt = f"### Query:\n{self.query}\n\n"
#         user_prompt += "### Documents:\n"
#         for i, document in enumerate(self.documents, 1):
#             self.key_to_document[i] = document.docid
#             user_prompt += f"Document {i}:\n" f"Text: {document.text}\n-----\n"

#         if self.use_COT:
#             user_prompt += (
#                 "\n\nPlease provide your analysis and relevance scores for the documents:\n"
#             )
#         else:
#             user_prompt += "\n\nPlease provide your relevance scores for the documents:\n"
#         return user_prompt


class BatchwiseRankRelevancePrompt(BasePromptFunction):
    def __init__(
        self,
        query: str,
        documents: list[SearchResult],
        use_COT: bool = False,
        num_anchor: int = 4,
        output_schema: Optional[BaseModel] = None,
        **kwargs,
    ):
        super(BatchwiseRankRelevancePrompt, self).__init__(**kwargs)
        self.query = query
        self.documents = documents
        self.use_COT = use_COT
        self.num_anchor = num_anchor
        self.output_schema = output_schema

        self.key_to_document = {}

    def system(self):
        prompt = (
            "You are a helpful assistant for document ranking. You will be given a query and a list of documents.\n"
            "Your task is to output a relevance score (R) between 0 and 100 for each document. "
            "This score should reflect both how well the document matches the query and how it compares to other documents in the list. "
            "A score of 0 means the document is the least relevant among all documents for this query, "
            "80 might mean that it could be the most relevant among the documents but not exactly relevant to the query, "
            "while 100 means it's the most relevant relative to other documents and exactly to the query.\n\n"
            "Scoring guidelines:\n"
            "- 0: The document is completely irrelevant to the query, containing no meaningful or useful information related to it. It is the least relevant document among all presented.\n"
            "- 1-20: The document has a minimal connection to the query, possibly mentioning relevant terms but largely out of context. Content may be tangential or incidental, lacking substantive connection to the query.\n"
            "- 21-40: The document provides some relevance to the query by briefly addressing a related topic, but it lacks depth or specificity. Information may be too general or incomplete to be considered genuinely useful.\n"
            "- 41-60: The document is moderately relevant, addressing the query to some extent. It covers several key aspects but may lack depth or overlook certain important points, providing a partial view rather than comprehensive coverage.\n"
            "- 61-80: The document is highly relevant and covers the query well, though it may fall short of being the absolute best match. It is informative and closely aligned with the query, but may lack exceptional insight or miss minor details.\n"
            "- 81-99: The document is extremely relevant and nearly perfect in addressing the query, covering all essential aspects with minor imperfections. It’s among the top matches and provides substantial detail and context.\n"
            "- 100: The document is the perfect match for the query, addressing all aspects comprehensively and precisely. It is the most relevant document in the list, with exceptional detail and alignment with the query.\n\n"
            "Consider both absolute relevance to the query and relative relevance compared to other documents.\n"
        )
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
            prompt += f"To output a list of relevance scores for each document, you need to follow the following schema: {self.output_schema.model_json_schema()}\n"
        return prompt

    def user(self):
        user_prompt = f"### Query:\n{self.query}\n\n"
        user_prompt += "### Documents:\n"
        for i, document in enumerate(self.documents, 1):
            self.key_to_document[i] = document.docid
            user_prompt += f"Document {i}:\n" f"Text: {document.text}\n-----\n"

        if self.use_COT:
            user_prompt += (
                "\n\nPlease provide your analysis and relevance scores for the documents:\n"
            )
        else:
            user_prompt += "\n\nPlease provide your relevance scores for the documents:\n"
        return user_prompt


class Anchor(BaseModel):
    docid: str
    text: str
    score: float
    reasoning: str = ""


class ChunkwiseRankRelevancePrompt(BasePromptFunction):
    def __init__(
        self,
        query: str,
        anchors: Optional[List[Anchor]],
        documents: list[SearchResult],
        use_COT: bool = False,
        output_schema: Optional[Union[BaseModel, Type[BaseModel]]] = None,
        **kwargs,
    ):
        # super(ChunkwiseRankRelevancePrompt, self).__init__(**kwargs)
        self.query = query
        self.anchors = anchors
        self.documents = documents
        self.use_COT = use_COT
        self.output_schema = output_schema

        self.num_anchor = len(anchors) if anchors else 0
        self.key_to_document = {}

    def system(self):
        prompt = (
            "You are a helpful assistant for document ranking. You will be given a query and a list of documents.\n"
            "Your task is to output a relevance score (R) between 0 and 100 for each document. "
            "This score should reflect both how well the document matches the query and how it compares to other documents in the list. "
            "A score of 0 means the document is the least relevant among all documents for this query, "
            "80 might mean that it could be the most relevant among the documents but not exactly relevant to the query, "
            "while 100 means it's the most relevant relative to other documents and exactly to the query.\n\n"
            "Scoring guidelines:\n"
            "- 0: The document is completely irrelevant to the query, containing no meaningful or useful information related to it. It is the least relevant document among all presented.\n"
            "- 1-20: The document has a minimal connection to the query, possibly mentioning relevant terms but largely out of context. Content may be tangential or incidental, lacking substantive connection to the query.\n"
            "- 21-40: The document provides some relevance to the query by briefly addressing a related topic, but it lacks depth or specificity. Information may be too general or incomplete to be considered genuinely useful.\n"
            "- 41-60: The document is moderately relevant, addressing the query to some extent. It covers several key aspects but may lack depth or overlook certain important points, providing a partial view rather than comprehensive coverage.\n"
            "- 61-80: The document is highly relevant and covers the query well, though it may fall short of being the absolute best match. It is informative and closely aligned with the query, but may lack exceptional insight or miss minor details.\n"
            "- 81-99: The document is extremely relevant and nearly perfect in addressing the query, covering all essential aspects with minor imperfections. It’s among the top matches and provides substantial detail and context.\n"
            "- 100: The document is the perfect match for the query, addressing all aspects comprehensively and precisely. It is the most relevant document in the list, with exceptional detail and alignment with the query.\n\n"
            "Consider both absolute relevance to the query and relative relevance compared to other documents.\n"
        )
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
            prompt += "Output a list of relevance scores for each document.\n"

        prompt += f"To output a list of relevance scores for each document, you need to follow the following schema: {self.output_schema.model_json_schema()}\n"
        return prompt

    def user(self):
        if self.anchors:
            user_prompt = f"Give a relevance score (R) between 0 and 100 for each document based on the query and take the anchors as reference.\n\n"
        else:
            user_prompt = f"Give a relevance score (R) between 0 and 100 for each document based on the query.\n\n"

        user_prompt += f"### Query:\n{self.query}\n\n"

        if self.anchors:
            user_prompt += "### Anchors:\n"
            for anchor in self.anchors:
                i = len(self.key_to_document) + 1
                self.key_to_document[i] = anchor.docid
                user_prompt += (
                    f"Anchor {i}:\n" f"Text: {anchor.text}\n" f"Relevance score: {anchor.score}"
                )
                if anchor.reasoning:
                    user_prompt += f"\nReasoning: {anchor.reasoning}"
                user_prompt += "\n-----\n"
            user_prompt += "\n"

        user_prompt += "### Documents:\n"
        for document in self.documents:
            i = len(self.key_to_document) + 1
            self.key_to_document[i] = document.docid
            user_prompt += f"Document id: {i}\n" f"Text: {document.text}\n-----\n"

        if self.use_COT:
            user_prompt += (
                "\n\nPlease provide your analysis and relevance scores for the documents:\n"
            )
        else:
            user_prompt += "\n\nPlease provide your relevance scores for the documents:\n"
        return user_prompt


class DocumentRelevanceScoresBatchResponse(BaseModel):
    class DocumentRelevanceScore(BaseModel):
        document_id: int = Field(
            description="The document number",
        )
        relevance_score: float = Field(
            description="The relevance score of the corresponding document relative to the query",
        )

    document_scores: list[DocumentRelevanceScore] = Field(
        description="The list of scoring results for the documents",
        default_factory=list,
    )


if __name__ == "__main__":
    query = "What is the capital of France?"
    documents = [
        SearchResult(docid="d1", score=0.5, text="Paris is the capital of France."),
        SearchResult(docid="d2", score=0.3, text="The Eiffel Tower is located in Paris."),
        SearchResult(docid="d3", score=0.2, text="France is known for its cuisine."),
    ]
    # prompt = SetwiseRankRelevancePrompt(query, documents, use_COT=False)
    # print(f"### System Prompt:\n{prompt.system()}")
    # print(f"### User Prompt:\n{prompt.user()}")
    # print(prompt.key_to_document)

    print("### Batchwise Prompt")
    prompt = BatchwiseRankRelevancePrompt(
        query, documents, use_COT=False, output_schema=DocumentRelevanceScoresBatchResponse
    )
    print(f"### System Prompt:\n{prompt.system()}")
    print(f"### User Prompt:\n{prompt.user()}")
    print(prompt.key_to_document)

    print("### Chunkwise Prompt")
    anchors = [
        Anchor(
            docid="a1",
            text="Paris is the capital of France.",
            score=100,
            reasoning="Paris is the capital of France.",
        ),
        Anchor(
            docid="a2",
            text="The Eiffel Tower is located in Paris.",
            score=80,
            reasoning="The Eiffel Tower is a landmark in Paris, not necessarily the capital.",
        ),
    ]
    prompt = ChunkwiseRankRelevancePrompt(
        query,
        anchors=None,
        documents=documents,
        use_COT=True,
        output_schema=DocumentRelevanceScoresBatchResponse,
    )
    print(f"### System Prompt:\n{prompt.system()}")
    print(f"### User Prompt:\n{prompt.user()}")
    print(prompt.key_to_document)
