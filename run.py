import logging
import ir_datasets
from pyserini.search.lucene import LuceneSearcher
from pyserini.search._base import get_topics
from sympy import use
from llmrankers.batchrankers import OpenAiBatchRanker, OpenaiBatchRanker
from llmrankers.rankers import SearchResult
from llmrankers.pointwise import PointwiseLlmRanker, MonoT5LlmRanker, OpenAIPointwiseLlmRanker
from llmrankers.setwise import SetwiseLlmRanker, OpenAISetwiseLlmRanker
from llmrankers.pairwise import (
    PairwiseLlmRanker,
    DuoT5LlmRanker,
    OpenAIPairwiseLlmRanker,
)
from llmrankers.listwise import OpenAIListwiseLlmRanker, ListwiseLlmRanker
from llmrankers.tourwise import TourwiseLlmRanker
from beir.retrieval.evaluation import EvaluateRetrieval
from beir import LoggingHandler
from tqdm import tqdm
import argparse
import sys
import json
import time
import random
import os
from eval import load_qrels_for_evaluation

from llmrankers.chunkrankers import OpenaiChunkRanker, VLLMChunkRanker
import asyncio

# convert to price
from llmrankers.utils.pricing import get_pricing


random.seed(929)
logger = logging.getLogger(__name__)

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args(parser, commands):
    # Divide argv by commands
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)
    # Initialize namespace
    args = argparse.Namespace()
    for c in commands.choices:
        setattr(args, c, None)
    # Parse each command
    parser.parse_args(split_argv[0], namespace=args)  # Without command
    for argv in split_argv[1:]:  # Commands
        n = argparse.Namespace()
        setattr(args, argv[0], n)
        parser.parse_args(argv, namespace=n)
    return args


def write_run_file(path, results, tag):
    with open(path, "w") as f:
        for qid, _, ranking in results:
            rank = 1
            for doc in ranking:
                docid = doc.docid
                score = doc.score
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score}\t{tag}\n")
                rank += 1


def main(args):

    if args.pointwise:
        if args.run.openai_key:
            ranker = OpenAIPointwiseLlmRanker(
                model_name_or_path=args.run.model_name_or_path,
                tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                device=args.run.device,
                api_key=args.run.openai_key,
                method=args.pointwise.method,
                batch_size=args.pointwise.batch_size,
            )
        elif "monot5" in args.run.model_name_or_path:
            ranker = MonoT5LlmRanker(
                model_name_or_path=args.run.model_name_or_path,
                tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                device=args.run.device,
                cache_dir=args.run.cache_dir,
                method=args.pointwise.method,
                batch_size=args.pointwise.batch_size,
            )
        else:
            ranker = PointwiseLlmRanker(
                model_name_or_path=args.run.model_name_or_path,
                tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                device=args.run.device,
                cache_dir=args.run.cache_dir,
                method=args.pointwise.method,
                batch_size=args.pointwise.batch_size,
            )

    elif args.setwise:
        if args.run.openai_key:
            ranker = OpenAISetwiseLlmRanker(
                model_name_or_path=args.run.model_name_or_path,
                api_key=args.run.openai_key,
                num_child=args.setwise.num_child,
                method=args.setwise.method,
                k=args.setwise.k,
            )
        else:
            ranker = SetwiseLlmRanker(
                model_name_or_path=args.run.model_name_or_path,
                tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                device=args.run.device,
                cache_dir=args.run.cache_dir,
                num_child=args.setwise.num_child,
                scoring=args.run.scoring,
                method=args.setwise.method,
                num_permutation=args.setwise.num_permutation,
                k=args.setwise.k,
            )

    elif args.pairwise:
        if args.pairwise.method != "allpair":
            args.pairwise.batch_size = 2
            logger.info(f"Setting batch_size to 2.")

        if args.run.openai_key:
            ranker = OpenAIPairwiseLlmRanker(
                model_name_or_path=args.run.model_name_or_path,
                api_key=args.run.openai_key,
                method=args.pairwise.method,
                k=args.pairwise.k,
            )

        elif "duot5" in args.run.model_name_or_path:
            ranker = DuoT5LlmRanker(
                model_name_or_path=args.run.model_name_or_path,
                tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                device=args.run.device,
                cache_dir=args.run.cache_dir,
                method=args.pairwise.method,
                batch_size=args.pairwise.batch_size,
                k=args.pairwise.k,
            )
        else:
            ranker = PairwiseLlmRanker(
                model_name_or_path=args.run.model_name_or_path,
                tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                device=args.run.device,
                cache_dir=args.run.cache_dir,
                method=args.pairwise.method,
                batch_size=args.pairwise.batch_size,
                k=args.pairwise.k,
            )

    elif args.listwise:
        if args.run.openai_key:
            ranker = OpenAIListwiseLlmRanker(
                model_name_or_path=args.run.model_name_or_path,
                api_key=args.run.openai_key,
                window_size=args.listwise.window_size,
                step_size=args.listwise.step_size,
                scoring=args.run.scoring,
                num_repeat=args.listwise.num_repeat,
            )
        else:
            ranker = ListwiseLlmRanker(
                model_name_or_path=args.run.model_name_or_path,
                tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                device=args.run.device,
                cache_dir=args.run.cache_dir,
                window_size=args.listwise.window_size,
                step_size=args.listwise.step_size,
                scoring=args.run.scoring,
                num_repeat=args.listwise.num_repeat,
            )

    elif args.batchwise:
        # ranker = OpenAiBatchRanker(
        # ranker = OpenaiBatchRanker(
        if args.batchwise.use_vllm:
            ranker = VLLMChunkRanker(
                model_name_or_path=args.run.model_name_or_path,
                base_url=args.batchwise.vllm_url,
                batch_size=args.batchwise.batch_size,
                num_vote=args.batchwise.num_vote,
                method=args.batchwise.method,
                temperature=args.batchwise.temperature,
                num_anchor=args.batchwise.num_anchor,
                use_COT=args.batchwise.use_COT,
                use_COT_anchor=args.batchwise.use_COT_anchor,
                guided_decoding_backend=args.batchwise.vllm_guided_decoding_backend,
            )
        else:
            ranker = OpenaiChunkRanker(
                num_anchor=args.batchwise.num_anchor,
                batch_size=args.batchwise.batch_size,
                num_vote=args.batchwise.num_vote,
                method=args.batchwise.method,
                model_name_or_path=args.run.model_name_or_path,
                temperature=args.batchwise.temperature,
                use_COT=args.batchwise.use_COT,
                use_COT_anchor=args.batchwise.use_COT_anchor,
            )

    elif args.tourwise:
        ranker = TourwiseLlmRanker(
            model_name_or_path=args.run.model_name_or_path,
            batch_size=args.tourwise.batch_size,
            num_tournaments=args.tourwise.num_tournaments,
            temperature=args.tourwise.temperature,
            api_key=args.run.openai_key,
        )
    else:
        raise ValueError("Must specify either --pointwise, --setwise, --pairwise, --listwise, --batchwise, or --tourwise.")

    print(f"Ranker: {ranker}")

    query_map = {}
    if args.run.ir_dataset_name is not None:
        dataset = ir_datasets.load(args.run.ir_dataset_name)
        for query in dataset.queries_iter():
            qid = query.query_id
            text = query.text
            query_map[qid] = ranker.truncate(text, args.run.query_length)
        dataset = ir_datasets.load(args.run.ir_dataset_name)
        docstore = dataset.docs_store()
    else:
        topics = get_topics(args.run.pyserini_index + "-test")
        for topic_id in list(topics.keys()):
            text = topics[topic_id]["title"]
            query_map[str(topic_id)] = ranker.truncate(text, args.run.query_length)
        docstore = LuceneSearcher.from_prebuilt_index(args.run.pyserini_index + ".flat")

    run_stats = {}
    if not args.run.skip_rerank:
        logger.info(f"Loading first stage run from {args.run.run_path}.")
        first_stage_rankings = []
        with open(args.run.run_path, "r") as f:
            current_qid, current_ranking = None, []
            for line in tqdm(f):
                # 19335 Q0 8412684 1 10.606700 Anserini
                qid, _, docid, _, score, _ = line.strip().split()
                if qid != current_qid:
                    if current_qid is not None:
                        first_stage_rankings.append(
                            (current_qid, query_map[current_qid], current_ranking[: args.run.hits])
                        )
                    current_ranking = []
                    current_qid = qid
                if len(current_ranking) >= args.run.hits:
                    continue
                if args.run.ir_dataset_name is not None:
                    text = docstore.get(docid).text
                    if "title" in dir(docstore.get(docid)):
                        text = f"{docstore.get(docid).title} {text}"
                else:
                    data = json.loads(docstore.doc(docid).raw())
                    text = data["text"]
                    if "title" in data:
                        text = f'{data["title"]} {text}'
                text = ranker.truncate(text, args.run.passage_length)
                current_ranking.append(SearchResult(docid=docid, score=float(score), text=text))
            first_stage_rankings.append(
                (current_qid, query_map[current_qid], current_ranking[: args.run.hits])
            )

        reranked_results = []
        total_comparisons = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        tic = time.time()
        for qid, query, ranking in tqdm(first_stage_rankings):
            if args.run.shuffle_ranking is not None:
                if args.run.shuffle_ranking == "random":
                    random.shuffle(ranking)
                elif args.run.shuffle_ranking == "inverse":
                    ranking = ranking[::-1]
                else:
                    raise ValueError(f"Invalid shuffle ranking method: {args.run.shuffle_ranking}.")
            reranked_results.append((qid, query, ranker.rerank(query, ranking)))
            total_comparisons += ranker.total_compare
            total_prompt_tokens += ranker.total_prompt_tokens
            total_completion_tokens += ranker.total_completion_tokens

            print(f"Current total prompt tokens: {total_prompt_tokens:,}")
            print(f"Current total completion tokens: {total_completion_tokens:,}")
            if "gpt" in args.run.model_name_or_path:
                prices = get_pricing(
                    args.run.model_name_or_path, total_prompt_tokens, total_completion_tokens
                )
                avg_prices_per_query = {k: v / len(reranked_results) for k, v in prices.items()}
                estimated_total_cost = {
                    k: v * len(first_stage_rankings) for k, v in avg_prices_per_query.items()
                }
                prices = {k: f"{v:.4f}$" for k, v in prices.items()}
                avg_prices_per_query = {k: f"{v:.4f}$" for k, v in avg_prices_per_query.items()}
                estimated_total_cost = {k: f"{v:.4f}$" for k, v in estimated_total_cost.items()}

                print(f"Total prices: {prices}")
                print(f"Avg prices per query: {avg_prices_per_query}")
                print(f"Estimated total cost: {estimated_total_cost}")

        toc = time.time()
        if args.batchwise:
            toc -= ranker.delay_per_query * len(reranked_results)  # delay for each query
            toc -= ranker.extra_time_for_asyncio  # extra time for handling asyncio

        print(f"Number of reranked queries: {len(reranked_results)}")
        print(f"total prompt tokens: {total_prompt_tokens}")
        print(f"total completion tokens: {total_completion_tokens}")
        print(f"Avg comparisons: {total_comparisons/len(reranked_results)}")
        print(f"Avg prompt tokens: {total_prompt_tokens/len(reranked_results)}")
        print(f"Avg completion tokens: {total_completion_tokens/len(reranked_results)}")
        print(f"Avg time per query: {(toc-tic)/len(reranked_results)}")

        run_stats = {
            "n_queries": len(reranked_results),
            "num_prompt_tokens": total_prompt_tokens,
            "num_completion_tokens": total_completion_tokens,
            "num_comparisons": total_comparisons,
            "avg_prompt_tokens": total_prompt_tokens / len(reranked_results),
            "avg_completion_tokens": total_completion_tokens / len(reranked_results),
            "avg_comparisons": total_comparisons / len(reranked_results),
            "avg_time_per_query": (toc - tic) / len(reranked_results),
        }
        if "gpt" in args.run.model_name_or_path:
            prices = get_pricing(
                args.run.model_name_or_path, total_prompt_tokens, total_completion_tokens
            )
            avg_prices_per_query = {k: v / len(reranked_results) for k, v in prices.items()}

            prices = {k: f"{v:.4f}$" for k, v in prices.items()}
            avg_prices_per_query = {k: f"{v:.4f}$" for k, v in avg_prices_per_query.items()}

            run_stats["prices"] = prices
            run_stats["avg_prices_per_query"] = avg_prices_per_query

        os.makedirs(os.path.dirname(args.run.save_path), exist_ok=True)
        write_run_file(args.run.save_path, reranked_results, "LLMRankers")

    if args.eval:
        rerank_result_path = args.run.save_path
        dataset_name = args.eval.dataset_name or args.run.ir_dataset_name
        k_values = [int(k) for k in args.eval.k_values.split(",")]

        rerank_results_for_eval = {}
        for line in open(rerank_result_path, "r"):
            query_id, _, doc_id, rank, score, _ = line.strip().split("\t")
            if query_id not in rerank_results_for_eval:
                rerank_results_for_eval[query_id] = {}
            rerank_results_for_eval[query_id][doc_id] = float(score)

        query_ids = list(rerank_results_for_eval.keys())
        qrels = load_qrels_for_evaluation(dataset_name, query_ids)
        # for qid, _, ranking in reranked_results:
        #     rerank_results_for_eval[qid] = {}
        #     for doc in ranking:
        #         docid = doc.docid
        #         score = doc.score
        #         rerank_results_for_eval[qid][docid] = float(score)

        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            qrels, rerank_results_for_eval, k_values=k_values
        )
        try:  # Error on TREC-DL 2020
            mrr = EvaluateRetrieval.evaluate_custom(
                qrels, rerank_results_for_eval, metric="mrr", k_values=k_values
            )
        except Exception as e:
            mrr = None
            logger.warning(f"Error calculating MRR: {e}")

        eval_output_file = args.run.save_path.replace(".txt", ".json")
        if os.path.exists(eval_output_file):
            eval_results = json.load(open(eval_output_file, "r"))
            logger.info(f"Loaded existing evaluation results from {eval_output_file}.")
        else:
            eval_results = {}
            logger.info(f"New evaluation results file will be written to {eval_output_file}.")
        eval_results.update(
            {
                "NDCG": ndcg,
                "MAP": _map,
                "Recall": recall,
                "Precision": precision,
                "MRR": mrr,
                **run_stats,
            }
        )
        with open(eval_output_file, "w") as f:
            json.dump(eval_results, f, indent=4, ensure_ascii=False)
        logger.info(f"Written evaluation results to {eval_output_file}.")
    print("Done!")


async def stop():
    loop = asyncio.get_event_loop()
    loop.stop()
    loop.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(title="sub-commands")

    run_parser = commands.add_parser("run")
    run_parser.add_argument(
        "--run_path",
        type=str,
        help="Path to the first stage run file (TREC format) to rerank.",
    )
    run_parser.add_argument(
        "--save_path",
        type=str,
        help="Path to save the reranked run file (TREC format).",
    )
    run_parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to the pretrained model or model identifier from huggingface.co/models",
    )
    run_parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="Path to the pretrained tokenizer or tokenizer identifier from huggingface.co/tokenizers",
    )
    run_parser.add_argument("--ir_dataset_name", type=str, default=None)
    run_parser.add_argument("--pyserini_index", type=str, default=None)
    run_parser.add_argument("--hits", type=int, default=100)
    run_parser.add_argument("--query_length", type=int, default=128)
    run_parser.add_argument("--passage_length", type=int, default=128)
    run_parser.add_argument("--device", type=str, default="cuda")
    run_parser.add_argument("--cache_dir", type=str, default=None)
    run_parser.add_argument("--openai_key", type=str, default=None)
    run_parser.add_argument(
        "--scoring",
        type=str,
        default="generation",
        choices=["generation", "likelihood"],
    )
    run_parser.add_argument(
        "--shuffle_ranking", type=str, default=None, choices=["inverse", "random"]
    )
    run_parser.add_argument(
        "--skip_rerank",
        type=str2bool,
        default=False,
        help="Skip reranking to go to evaluation or the next step.",
    )

    # Pointwise reranking
    pointwise_parser = commands.add_parser("pointwise")
    pointwise_parser.add_argument("--method", type=str, default="yes_no", choices=["qlm", "yes_no"])
    pointwise_parser.add_argument("--batch_size", type=int, default=2)

    pairwise_parser = commands.add_parser("pairwise")
    pairwise_parser.add_argument(
        "--method",
        type=str,
        default="allpair",
        choices=["allpair", "heapsort", "bubblesort"],
    )
    pairwise_parser.add_argument("--batch_size", type=int, default=2)
    pairwise_parser.add_argument("--k", type=int, default=10)

    # Setwise reranking
    setwise_parser = commands.add_parser("setwise")
    setwise_parser.add_argument("--num_child", type=int, default=3)
    setwise_parser.add_argument(
        "--method", type=str, default="heapsort", choices=["heapsort", "bubblesort"]
    )
    setwise_parser.add_argument("--k", type=int, default=10)
    setwise_parser.add_argument("--num_permutation", type=int, default=1)

    # Listwise reranking
    listwise_parser = commands.add_parser("listwise")
    listwise_parser.add_argument("--window_size", type=int, default=3)
    listwise_parser.add_argument("--step_size", type=int, default=1)
    listwise_parser.add_argument("--num_repeat", type=int, default=1)

    # Batchwise reranking
    batchwise_parser = commands.add_parser("batchwise")
    batchwise_parser.add_argument("--num_anchor", type=int, default=4)
    batchwise_parser.add_argument("--batch_size", type=int, default=10)
    batchwise_parser.add_argument("--num_vote", type=int, default=5)
    batchwise_parser.add_argument(
        "--method", type=str, default="random", choices=["random", "top", "none"]
    )
    batchwise_parser.add_argument("--temperature", type=float, default=0.5)
    batchwise_parser.add_argument(
        "--use_COT", type=str2bool, default=True, help="Use Chain of Thought reasoning"
    )
    batchwise_parser.add_argument(
        "--use_COT_anchor",
        type=str2bool,
        default=True,
        help="Use Chain of Thought reasoning when ranking scores for anchors",
    )
    batchwise_parser.add_argument(
        "--use_vllm", type=str2bool, default=False, help="Use vLLM server"
    )
    batchwise_parser.add_argument("--vllm_url", type=str, default="http://0.0.0.0:8000/v1")
    batchwise_parser.add_argument("--vllm_guided_decoding_backend", type=str, default="outlines")

    # Tourwise reranking
    tourwise_parser = commands.add_parser("tourwise")
    tourwise_parser.add_argument("--batch_size", type=int, default=10)
    tourwise_parser.add_argument("--num_tournaments", type=int, default=10)
    tourwise_parser.add_argument("--temperature", type=float, default=0.5)

    # Evaluation
    eval_parser = commands.add_parser("eval")
    eval_parser.add_argument("--dataset_name", type=str)
    eval_parser.add_argument(
        "--k_values",
        type=str,
        default="3,5,10,25,50,100",
        help="Comma separated list of k values. Default: 3,5,10,25,50,100",
    )
    eval_parser.add_argument(
        "--eval_output_file",
        type=str,
        help="Path to save the evaluation results. If not specified, use the same path as the run file but the format is .json.",
    )

    args = parse_args(parser, commands)

    if args.run.ir_dataset_name is not None and args.run.pyserini_index is not None:
        raise ValueError("Must specify either --ir_dataset_name or --pyserini_index, not both.")

    if args.eval and (args.eval.dataset_name is None and args.eval.ir_dataset_name is None):
        raise ValueError(
            "Must specify --dataset_name in eval args or --ir_dataset_name in run args to run evaluation after reranking."
        )

    arg_dict = vars(args)
    if arg_dict["run"] is None or (
        sum(arg_dict[arg] is not None for arg in arg_dict) != 2 and "eval" not in arg_dict
    ):
        raise ValueError(
            "Need to set --run and can only set one of --pointwise, --pairwise, --setwise, --listwise, --batchwise, or --tourwise"
        )
    main(args)
