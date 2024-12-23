from typing import List, Dict
import json
import os
import random
import copy
import time
import numpy as np
import openai
from openai import OpenAI
from tqdm import tqdm
from multiprocessing import Process, Manager, Value
import logging
from .rankers import LlmRanker, SearchResult
import tiktoken


class TourrankLlmRanker(LlmRanker):
    def __init__(
        self,
        model_name_or_path: str = "gpt-3.5-turbo",
        batch_size: int = 10,
        num_tournaments: int = 10,
        temperature: float = 0.5,
        api_key: str = None,
    ):
        """Initialize the Tourwise LLM Ranker.

        Args:
            model_name_or_path (str): Name or path of the LLM model to use
            batch_size (int): Size of document batches for processing
            num_tournaments (int): Number of tournament iterations
            temperature (float): Temperature for LLM sampling
            api_key (str): OpenAI API key
        """
        self.model = model_name_or_path
        self.batch_size = batch_size
        self.num_tournaments = num_tournaments
        self.temperature = temperature

        # Encoder
        self.tokenizer = tiktoken.encoding_for_model(model_name_or_path)

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        # Tracking metrics
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

    def truncate(self, text, length):
        return self.tokenizer.decode(self.tokenizer.encode(text)[:length])

    def get_response(
        self, messages: List[Dict], total_compare, total_completion_tokens, total_prompt_tokens
    ) -> str:
        """Get response from LLM model.

        Args:
            messages (List[Dict]): List of message dictionaries for the conversation

        Returns:
            str: Model response text
        """
        try:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature, timeout=15
                )
                # add completion tokens to total
                total_compare.value = total_compare.value + 1
                total_completion_tokens.value += completion.usage.completion_tokens
                total_prompt_tokens.value += completion.usage.prompt_tokens
                return completion.choices[0].message.content
            except openai.InternalServerError:
                print("openai.InternalServerError")
                return self.get_response(messages)
        except openai.RateLimitError:
            print("Rate limit exceeded and trying again...")
            time.sleep(20)  # Add delay before retry
            return self.get_response(messages)

    def get_prefix_role_prompt(self, query: str, N: int, M: int) -> List[Dict]:
        """Get the prefix prompt for document comparison.

        Args:
            query (str): Search query
            N (int): Number of documents to compare
            M (int): Number of documents to select

        Returns:
            List[Dict]: Formatted prompt messages
        """
        return [
            {
                "role": "system",
                "content": "You are an intelligent assistant that can compare multiple documents based on their relevancy to the given query.",
            },
            {
                "role": "user",
                "content": f"I will provide you with the given query and {N} documents. \nConsider the content of all the documents comprehensively and select the {M} documents that are most relevant to the given query: {query}.",
            },
            {"role": "assistant", "content": "Okay, please provide the documents."},
        ]

    def get_post_role_prompt(self, query: str, M: int) -> str:
        """Get the post-processing prompt.

        Args:
            query (str): Search query
            M (int): Number of documents to select

        Returns:
            str: Formatted prompt text
        """
        return f"""The Query is: {query}.
Now, you must output the top {M} documents that are most relevant to the Query using the following format strictly, and nothing else. Don't output any explanation, just the following format:
Document 3, ..., Document 1"""

    def get_top_M(
        self, answer: str, N: int = 10, M: int = 5, groups_docid: List[str] = []
    ) -> List[str]:
        """Extract top M document IDs from model answer.

        Args:
            answer (str): Model response text
            N (int): Total number of documents
            M (int): Number of documents to select
            groups_docid (List[str]): List of document IDs

        Returns:
            List[str]: Selected document IDs
        """
        temp = answer.split("\n")
        temp_length = len(temp)
        for i in range(1, temp_length + 1):
            if "Document" in temp[i * (-1)]:
                temp = temp[i * (-1)]
                break
        temp = temp.split(":")[-1]
        temp = temp.split(".")[0]
        temp = temp.split(",")
        top_M = []
        for doc in temp:
            try:
                try:
                    flag = 0
                    if "..." in doc:
                        flag = 1
                    if flag == 0:
                        doc_num = int(doc.split()[-1]) - 1
                        top_M.append(doc_num)
                except IndexError:
                    print("IndexError occurred in doc. (get_top_M), just ignore it.")
                    print(doc)
                    s = f"IndexError, {N}, {M}, {answer}"
                    debug_path = "./debug.txt"
                    with open(debug_path, "a") as f:
                        f.write("New Error: \n")
                        f.write(f"{top_M}\n")
                        f.write(f"{doc}\n")
                        f.write(f"{s}\n")
                        f.write("end\n\n")

            except ValueError:
                for j in range(1, N + 1):
                    if j not in top_M:
                        doc_num = j
                        break
                print("ValueError occurred in score. (get_top_M)")
                top_M.append(doc_num)
                s = f"ValueError, {N}, {M}, {answer}"
                debug_path = "./debug.txt"
                with open(debug_path, "a") as f:
                    f.write("New Error: \n")
                    f.write(f"{top_M}\n")
                    f.write(f"{doc}\n")
                    f.write(f"{s}\n")
                    f.write("end\n\n")

        top_M_ids = []
        for doc_num in top_M:
            top_M_ids.append(groups_docid[doc_num])

        return top_M_ids

    def filter_processing(
        self,
        y_it: int,
        query: str,
        docs_id: List[str],
        all_contents: Dict[str, str],
        docs_score_dicts_list: List,
        total_compare,
        total_completion_tokens,
        total_prompt_tokens,
    ) -> None:
        """Process one tournament iteration.

        Args:
            y_it (int): Tournament iteration number
            query (str): Search query
            docs_id (List[str]): List of document IDs
            all_contents (Dict[str, str]): Document contents mapping
            docs_score_dicts_list (List): Shared list for storing scores
        """
        # Initialize scores for this tournament
        docs_score_dict = {doc: 0 for doc in docs_id}

        # Stage 1: 100->50 documents
        N, M = 20, 10
        stage1_docs_id = docs_id
        docs_groups = self.get_groups_skip(stage1_docs_id, to_n_groups=5, m_docs_per_group=N)

        with Manager() as manager:
            groups_score_dict_list = manager.list()
            processes = []
            for groups in docs_groups:
                p = Process(
                    target=self.group_processing,
                    args=(
                        groups,
                        query,
                        N,
                        M,
                        all_contents,
                        groups_score_dict_list,
                        total_compare,
                        total_completion_tokens,
                        total_prompt_tokens,
                    ),
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            for group_score_dict in groups_score_dict_list:
                for doc, score in group_score_dict.items():
                    docs_score_dict[doc] += score

        # Get ranked list after stage 1
        ranked_list = self.sort_docs_by_relevance(
            list(docs_score_dict.keys()), list(docs_score_dict.values())
        )

        # Stage 2: 50->20 documents
        N, M = 10, 4
        stage2_docs_id = ranked_list[:50]
        docs_groups = self.get_groups_skip(stage2_docs_id, to_n_groups=5, m_docs_per_group=N)

        with Manager() as manager:
            groups_score_dict_list = manager.list()
            processes = []
            for groups in docs_groups:
                p = Process(
                    target=self.group_processing,
                    args=(
                        groups,
                        query,
                        N,
                        M,
                        all_contents,
                        groups_score_dict_list,
                        total_compare,
                        total_completion_tokens,
                        total_prompt_tokens,
                    ),
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            for group_score_dict in groups_score_dict_list:
                for doc, score in group_score_dict.items():
                    docs_score_dict[doc] += score

        ranked_list = self.sort_docs_by_relevance(
            list(docs_score_dict.keys()), list(docs_score_dict.values())
        )

        # Stage 3: 20->10 documents
        N, M = 20, 10
        stage3_docs_id = ranked_list[:20]
        docs_groups = self.get_groups_skip(stage3_docs_id, to_n_groups=1, m_docs_per_group=N)
        for groups in docs_groups:
            random.shuffle(groups)
            messages = self.get_prefix_role_prompt(query, N, M)
            for j, doc_id in enumerate(groups):
                content = all_contents[doc_id]
                messages.append({"role": "user", "content": f"Document {j+1}: {content}"})
                messages.append({"role": "assistant", "content": f"Received Document {j+1}."})
            messages.append({"role": "user", "content": self.get_post_role_prompt(query, M)})
            answer = self.get_response(
                messages, total_compare, total_completion_tokens, total_prompt_tokens
            )
            top_M_ids = self.get_top_M(answer, N=N, M=M, groups_docid=groups)
            for doc_id in top_M_ids:
                docs_score_dict[doc_id] += 1

        ranked_list = self.sort_docs_by_relevance(
            list(docs_score_dict.keys()), list(docs_score_dict.values())
        )

        # Stage 4: 10->5 documents
        N, M = 10, 5
        stage4_docs_id = ranked_list[:10]
        docs_groups = self.get_groups_skip(stage4_docs_id, to_n_groups=1, m_docs_per_group=N)
        for groups in docs_groups:
            random.shuffle(groups)
            messages = self.get_prefix_role_prompt(query, N, M)
            for j, doc_id in enumerate(groups):
                content = all_contents[doc_id]
                messages.append({"role": "user", "content": f"Document {j+1}: {content}"})
                messages.append({"role": "assistant", "content": f"Received Document {j+1}."})
            messages.append({"role": "user", "content": self.get_post_role_prompt(query, M)})
            answer = self.get_response(
                messages, total_compare, total_completion_tokens, total_prompt_tokens
            )
            top_M_ids = self.get_top_M(answer, N=N, M=M, groups_docid=groups)
            for doc_id in top_M_ids:
                docs_score_dict[doc_id] += 1

        ranked_list = self.sort_docs_by_relevance(
            list(docs_score_dict.keys()), list(docs_score_dict.values())
        )

        # Stage 5: 5->2 documents
        N, M = 5, 2
        stage5_docs_id = ranked_list[:5]
        docs_groups = self.get_groups_skip(stage5_docs_id, to_n_groups=1, m_docs_per_group=N)
        for groups in docs_groups:
            random.shuffle(groups)
            messages = self.get_prefix_role_prompt(query, N, M)
            for j, doc_id in enumerate(groups):
                content = all_contents[doc_id]
                messages.append({"role": "user", "content": f"Document {j+1}: {content}"})
                messages.append({"role": "assistant", "content": f"Received Document {j+1}."})
            messages.append({"role": "user", "content": self.get_post_role_prompt(query, M)})
            answer = self.get_response(
                messages, total_compare, total_completion_tokens, total_prompt_tokens
            )
            top_M_ids = self.get_top_M(answer, N=N, M=M, groups_docid=groups)
            for doc_id in top_M_ids:
                docs_score_dict[doc_id] += 1

        # Add final scores to shared list
        docs_score_dicts_list.append(docs_score_dict)
        print(f"Finished {y_it+1} process.")

    def group_processing(
        self,
        groups: List[str],
        query: str,
        N: int,
        M: int,
        all_contents: Dict[str, str],
        groups_score_dict_list: List,
        total_compare,
        total_completion_tokens,
        total_prompt_tokens,
    ) -> None:
        """Process a group of documents.

        Args:
            groups (List[str]): List of document IDs in the group
            query (str): Search query
            N (int): Number of documents to compare
            M (int): Number of documents to select
            all_contents (Dict[str, str]): Document contents mapping
            groups_score_dict_list (List): Shared list for storing scores
        """
        group_score_dict = {}
        random.shuffle(groups)
        messages = self.get_prefix_role_prompt(query, N, M)
        for j, doc_id in enumerate(groups):
            content = all_contents[doc_id]
            messages.append({"role": "user", "content": f"Document {j+1}: {content}"})
            messages.append({"role": "assistant", "content": f"Received Document {j+1}."})
        messages.append({"role": "user", "content": self.get_post_role_prompt(query, M)})
        answer = self.get_response(
            messages, total_compare, total_completion_tokens, total_prompt_tokens
        )
        top_M_ids = self.get_top_M(answer, N=N, M=M, groups_docid=groups)
        for doc_id in top_M_ids:
            group_score_dict[doc_id] = 1
        groups_score_dict_list.append(group_score_dict)

    def get_groups_chunk(self, docs_id: List[str], N: int = 10) -> List[List[str]]:
        """Split documents into chunks.

        Args:
            docs_id (List[str]): List of document IDs
            N (int): Chunk size

        Returns:
            List[List[str]]: List of document ID chunks
        """
        doc_num = len(docs_id)
        docs_groups = []
        cur_num = 0
        while cur_num < doc_num:
            docs_groups.append(docs_id[cur_num : cur_num + N])
            cur_num += N
        return docs_groups

    def get_groups_skip(
        self, docs_id: List[str], to_n_groups: int = 10, m_docs_per_group: int = 10
    ) -> List[List[str]]:
        """Split documents into groups with skipping.

        Args:
            docs_id (List[str]): List of document IDs
            to_n_groups (int): Number of groups to create
            m_docs_per_group (int): Number of documents per group

        Returns:
            List[List[str]]: List of document ID groups
        """
        docs_groups = []
        for i in range(to_n_groups):
            cur_group = []
            for j in range(m_docs_per_group):
                cur_group.append(docs_id[j * to_n_groups + i])
            docs_groups.append(cur_group)
        return docs_groups

    def sort_docs_by_relevance(
        self, doc_ids: List[str], relevance_scores: List[float]
    ) -> List[str]:
        """Sort documents by relevance scores.

        Args:
            doc_ids (List[str]): List of document IDs
            relevance_scores (List[float]): List of relevance scores

        Returns:
            List[str]: Sorted list of document IDs
        """
        combined = list(zip(doc_ids, relevance_scores))
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in sorted_combined]

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        """Rerank documents using tournament-style comparisons.

        Args:
            query (str): Search query
            ranking (List[SearchResult]): Initial document ranking

        Returns:
            List[SearchResult]: Reranked documents
        """
        # Reset metrics

        # Extract document IDs and contents
        docs_id = [doc.docid for doc in ranking]
        all_contents = {doc.docid: doc.text for doc in ranking}

        # Initialize scores
        docs_score_dict = {doc: 0 for doc in docs_id}

        # Run tournaments
        with Manager() as manager:
            total_compare = manager.Value("total_compare", 0)
            total_completion_tokens = manager.Value("total_completion_tokens", 0)
            total_prompt_tokens = manager.Value("total_prompt_tokens", 0)
            docs_score_dicts_list = manager.list()
            processes = []
            for y in range(self.num_tournaments):
                p = Process(
                    target=self.filter_processing,
                    args=(
                        y,
                        query,
                        docs_id,
                        all_contents,
                        docs_score_dicts_list,
                        total_compare,
                        total_completion_tokens,
                        total_prompt_tokens,
                    ),
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            self.total_compare = total_compare.value
            self.total_completion_tokens = total_completion_tokens.value
            self.total_prompt_tokens = total_prompt_tokens.value

            # Handle missing tournaments
            if self.num_tournaments - len(docs_score_dicts_list) > 0:
                for i in range(min(3, self.num_tournaments - len(docs_score_dicts_list))):
                    docs_score_dicts_list.append(docs_score_dicts_list[i])

            # Combine scores from all tournaments
            for tournament_scores in docs_score_dicts_list:
                for doc_id, score in tournament_scores.items():
                    docs_score_dict[doc_id] += score

        # Create final ranking
        ranked_list = self.sort_docs_by_relevance(
            list(docs_score_dict.keys()), list(docs_score_dict.values())
        )
        return [
            SearchResult(docid=doc_id, score=docs_score_dict[doc_id], text=all_contents[doc_id])
            for doc_id in ranked_list
        ]
