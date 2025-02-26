import os
import sys
import math
import json
import argparse
import random
import time
import torch
import openai

import numpy as np
import torch.nn.functional as F

from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch.nn as nn

from llmrankers.test_rankers.NewRanker10.NewRanker10 import OpenaiNewRanker10
from llmrankers.test_rankers.NewRanker10.prompt10 import SearchResult

from beir.retrieval.evaluation import EvaluateRetrieval


class Logger(object):

    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, "w")
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=""):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append("%s %.6f" % (key, np.mean(vals)))
        msg = "\n".join(msgs)
        self.log_file.write(msg + "\n")
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(str(msg) + "\n")
        self.log_file.flush()
        print(msg)


class PolicyNetwork(nn.Module):

    def __init__(
        self,
        model_config="bert-base-uncased",
        add_linear=False,
        embedding_size=128,
        freeze_encoder=True,
        cache_dir="./cache",
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_config, cache_dir=cache_dir)
        print("model_config:", model_config)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_config, cache_dir=cache_dir
        )

        # Freeze transformer encoder and only train the linear layer
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        if add_linear:
            # Add an additional small, adjustable linear layer on top of BERT tuned through RL
            self.embedding_size = embedding_size
            self.linear = nn.Linear(
                self.model.config.hidden_size, embedding_size
            )  # 768 for bert-base-uncased, distilbert-base-uncased
        else:
            self.linear = None

    def forward(self, input_list):
        input = self.tokenizer(input_list, truncation=True, padding=True, return_tensors="pt").to(
            self.model.device
        )
        # print(f"input: {input}")
        output = self.model(**input, output_hidden_states=True)
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]
        # Get [CLS] hidden states
        sentence_embedding = last_hidden_states[:, 0, :]  # len(input_list) x hidden_size
        # print(f"sentence_embedding: {sentence_embedding}")

        if self.linear:
            sentence_embedding = self.linear(sentence_embedding)  # len(input_list) x embedding_size

        return sentence_embedding


def load_data(args):
    with open("resources/rl/msmacro.json", "r") as f:
        data = json.load(f)

    ranking = data["ranking"]
    qrels = data["qrels"]

    # 80% for training, 20% for testing
    num_trains = int(len(ranking) * 0.8)
    train_ranking = [item for item in ranking[:num_trains]]
    test_ranking = [item for item in ranking[num_trains:]]

    return ranking, qrels, train_ranking, test_ranking


# @lru_cache(maxsize=10000)
def ranker_rerank(ranker, query, ranking, selected_anchors):
    ranker.delay_per_query = 0
    ranking_results = ranker.rerank(query, ranking, selected_anchors=selected_anchors)
    return ranking_results


def get_batch_reward_loss(scores, batch, batch_qids, batch_queries, batch_ranking, qrels):
    total_batch_loss = 0
    total_batch_reward = 0

    batch_losses = []
    batch_rewards = []
    batch_log_probs = []

    ## loop over each query in the batch
    for i in range(len(scores)):
        cur_qid = batch_qids[i]
        cur_query = batch_queries[i]
        cur_ranking = batch_ranking[i]
        cur_ranking = [
            SearchResult(docid=a["docid"], score=a["score"], text=a["text"]) for a in cur_ranking
        ]

        # interact with the environment to get rewards, which in our case is to feed the prompt into GPT-3 and evaluate the prediction
        anchor_prob = scores[i, :].clone().detach()
        anchor_prob = anchor_prob.cpu().numpy()
        anchor_prob = np.nan_to_num(anchor_prob, nan=0.000001)  # replace np.nan with 0
        # anchor_prob /= anchor_prob.sum()  # make probabilities sum to 1
        # print(f"anchor_prob: {anchor_prob}")

        # sample anchor ids from the distribution
        # anchor_ids = np.random.choice(
        #     range(len(anchor_prob)), args.shot_number, p=anchor_prob, replace=False
        # )
        import copy

        sorted_ranking = copy.deepcopy(cur_ranking)
        sorted_ranking = sorted(sorted_ranking, key=lambda x: x.score, reverse=True)
        interval_size = len(cur_ranking) // args.shot_number
        print(len(cur_ranking), interval_size)
        anchor_ids = []
        for idx in range(0, args.shot_number):
            start = idx * interval_size
            end = min(start + interval_size, len(ranking))
            interval_ids = list(range(start, end))
            interval_probs = anchor_prob[start:end]
            interval_probs = interval_probs / interval_probs.sum()
            anchor_id = np.random.choice(interval_ids, p=interval_probs, replace=False)
            anchor_ids.append(anchor_id)
        print(f"anchor_ids: {anchor_ids}")

        selected_anchors = [cur_ranking[i] for i in anchor_ids]

        # Define the ranker
        ranker = OpenaiNewRanker10(
            model_name_or_path="gpt-4o-mini",
            batch_size=8,
            num_vote=1,
            rerank_top_n=0,  # No rerank stage 2
        )

        # Convert docs from dict to SearchResult dataclass object
        ranker.delay_per_query = 0
        ranking_results = ranker_rerank(ranker, cur_query, cur_ranking, selected_anchors)

        log_prob = 0
        for anchor_id in anchor_ids:
            log_prob += torch.log(scores[i, anchor_id])
        # print(f"log_prob: {log_prob}")

        # Reward is ndcg@10
        ranking_results_for_eval = {cur_qid: {}}
        for d in ranking_results:
            ranking_results_for_eval[cur_qid][d.docid] = float(d.score)
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            qrels, ranking_results_for_eval, k_values=[10]
        )
        _reward = ndcg.get("NDCG@10", 0.0)

        # batch_reward += _reward
        # batch_loss -= _reward * log_prob
        # total_batch_reward += _reward
        # total_batch_loss -= _reward * log_prob

        batch_losses.append(-_reward * log_prob)
        batch_rewards.append(_reward)
        batch_log_probs.append(log_prob)

    print(f"Batch Rewards: {batch_rewards}")
    for i in range(len(batch_rewards)):
        # Do not include the current reward
        if i == 0:
            cur_baseline = torch.tensor(batch_rewards[1:], requires_grad=False)
        elif i == len(batch_rewards) - 1:
            cur_baseline = torch.tensor(batch_rewards[:-1], requires_grad=False)
        else:
            cur_baseline = torch.tensor(
                batch_rewards[:i] + batch_rewards[i + 1 :], requires_grad=False
            )

        cur_baseline = cur_baseline.sum() / (len(batch_rewards) - 1)
        total_batch_loss += -batch_log_probs[i] * (batch_rewards[i] - cur_baseline)
        total_batch_reward += batch_rewards[i]

    return anchor_ids, total_batch_reward, total_batch_loss


def policy_gradient_train(policy_model, train_samples, qrels, batch_size, n_epochs):
    # REINFORCE
    # if os.path.exists(args.ckpt_path):
    #     print("!!! Model dir already exists. Consider load it instead of training again.")

    optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.lr)

    num_batch = math.ceil(len(train_samples) / batch_size)
    train_ranking = [item[2] for item in train_samples]

    reward_history = []
    loss_history = []

    total_reward_history = []  # epoch based
    total_loss_history = []  # epoch based

    STOP_FLAG = False

    for epoch in range(n_epochs):
        logger.write(f"Epoch: {epoch}")

        total_train_reward = 0
        total_train_loss = 0

        # We can simply set the batch_size to len(train_data) in few-shot setting.
        for batch_i in range(num_batch):
            logger.write(f"Batch: {batch_i}")
            batch = train_samples[batch_i * batch_size : (batch_i + 1) * batch_size]
            batch_qids = [item[0] for item in batch]
            batch_queries = [item[1] for item in batch]
            batch_ranking = [item[2] for item in batch]
            batch_documents = [[d["text"] for d in item] for item in batch_ranking]

            # We need to encode queries and documents again every time we update the network
            embedding_queries = policy_model(batch_queries)  # n_queries x hidden_size
            embedding_docs = []
            for texts in batch_documents:
                # Embedding 100 documents per query ==> Extensive GPU consumption, should implement inner batching ^^ but I am too lazy right now
                embedding_docs.append(policy_model(texts))
            embedding_docs = torch.stack(embedding_docs)  # n_queries x 100 x hidden_size

            scores = torch.bmm(
                embedding_queries.unsqueeze(1), embedding_docs.transpose(1, 2)
            ).squeeze(1)
            # scores has shape: n_queries x 100

            scores = F.softmax(scores, dim=1)  # len(train_batch) x len(cand_examples)

            anchor_ids, reward, loss = get_batch_reward_loss(
                scores, batch, batch_qids, batch_queries, batch_ranking, qrels
            )

            logger.write(f"anchor_ids for sample[-1] in batch: {anchor_ids}")
            logger.write(
                f"Anchor prob for sample[-1] in batch: {[round(x,5) for x in scores[-1, :].tolist()]}"
            )
            logger.write(f"### reward for the batch: {reward}")
            logger.write(f"### loss for the batch: {loss}\n")

            # linear layer has Weight and bias
            # prev_param = list(policy_model.linear.parameters())[0].clone()
            # print(f"prev_param: {prev_param.data}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for each iteration/batch
            total_train_reward += reward
            total_train_loss += loss.item()

            reward_history.append(reward)
            loss_history.append(loss.item())

            if np.isnan(loss.item()):
                STOP_FLAG = True
                break

        # for each epoch
        total_reward_history.append(total_train_reward)
        total_loss_history.append(total_train_loss)

        best_reward = max(total_reward_history)
        best_loss = min(total_loss_history)

        best_reward_epoch = total_reward_history.index(best_reward)
        best_loss_epoch = total_loss_history.index(best_loss)

        logger.write("============================================")
        logger.write(f"### Epoch: {epoch} / {args.epochs}")
        logger.write(
            f"### Total reward: {total_train_reward}, "
            + f"Total loss: {round(total_train_loss,5)}, "
            + f"Best reward: {best_reward} at epoch {best_reward_epoch}, "
            + f"Best loss: {round(best_loss, 5)} at epoch {best_loss_epoch}\n"
        )

        # save every epoch
        ckpt_file = os.path.join(args.ckpt_path, f"ckpt_{epoch}.pt")
        torch.save(policy_model.linear.state_dict(), ckpt_file)
        logger.write(f"saved the ckpt to {ckpt_file}")

        # save best epoch
        if epoch == best_reward_epoch:
            ckpt_file = os.path.join(args.ckpt_path, "ckpt_best_reward.pt")
            torch.save(policy_model.linear.state_dict(), ckpt_file)
            logger.write(f"saved the best reward ckpt to {ckpt_file}")

        if epoch == best_loss_epoch:
            ckpt_file = os.path.join(args.ckpt_path, "ckpt_best_loss.pt")
            torch.save(policy_model.linear.state_dict(), ckpt_file)
            logger.write(f"saved the best loss ckpt to {ckpt_file}")

        # save reward and loss history
        history = {
            "reward_history": reward_history,
            "loss_history": loss_history,
            "total_reward_history": total_reward_history,
            "total_loss_history": total_loss_history,
        }
        history_file = os.path.join(args.ckpt_path, "history.json")
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2, separators=(",", ": "))

        # print cache info
        # logger.write(ranker_rerank.cache_info())
        logger.write("============================================\n")

        if STOP_FLAG:
            break

    # save in the end
    ckpt_file = os.path.join(args.ckpt_path, "ckpt_final.pt")
    torch.save(policy_model.linear.state_dict(), ckpt_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../data/tabmwp")
    parser.add_argument("--model", type=str, default="gpt3_rl")
    parser.add_argument("--option_inds", type=list, default=["A", "B", "C", "D", "E", "F"])

    # User options
    parser.add_argument("--label", type=str, default="exp0")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="TQ-A",
        choices=[
            "T-A",
            "Q-A",
            "Q-AS",
            "Q-SA",
            "TQ-A",
            "TQ-AS",
            "TQ-SA",
            "QT-A",
            "QT-AS",
            "QT-SA",
            "QTS-A",
            "TQS-A",
        ],
        help="prompt format template",
    )
    parser.add_argument(
        "--shot_number", type=int, default=4, help="Number of n-shot training examples."
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")

    # GPT-3 settings
    parser.add_argument(
        "--engine", type=str, default="text-davinci-002", choices=["text-davinci-002", "ada"]
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="The maximum number of tokens allowed for the generated answer.",
    )
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)

    # Policy gradient settings
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--model_config",
        type=str,
        default="bert-base-uncased",
        choices=["distilbert-base-uncased", "bert-base-uncased"],
    )
    parser.add_argument("--train_number", type=int, default=20, help="Number of training samples.")
    parser.add_argument("--cand_number", type=int, default=10, help="Number of candidate prompts.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate of policy network.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=128,
        help="Policy network final layer hidden state size.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Policy network training batch size. Set to train_number by default.",
    )
    parser.add_argument("--ckpt_root", type=str, default="../checkpoints")

    args = parser.parse_args()

    # print and save the args
    args.ckpt_path = os.path.join(args.ckpt_root, args.label)
    os.makedirs(args.ckpt_path, exist_ok=True)
    _logger = Logger(args.ckpt_path + "/args.txt")

    print("====Input Arguments====")
    _logger.write(json.dumps(vars(args), indent=2, sort_keys=False))

    return args


if __name__ == "__main__":

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # CPU random seed
    torch.cuda.manual_seed(args.seed)  # GPU random seed
    torch.backends.cudnn.benchmark = True

    ## problems, test question ids, candidate prompt pids, RL training pids
    # problems, cand_pids, train_pids = load_data(args)
    ranking, qrels, train_ranking, test_ranking = load_data(args)

    ## policy network
    policy_model = PolicyNetwork(
        model_config=args.model_config,
        add_linear=True,
        embedding_size=args.embedding_size,
        freeze_encoder=True,
    )

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")  # one GPU
    policy_model = policy_model.to(device)

    ## TRAINING
    logger = Logger(os.path.join(args.ckpt_path, "log.txt"))
    batch_size = args.batch_size
    n_epochs = args.epochs
    policy_gradient_train(policy_model, train_ranking, qrels, batch_size, n_epochs)
