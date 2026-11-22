"""LightGCN 测试（mid_data）：读各 seed 的最优模型，输出排行指标与 precision/recall/f1。

只读训练产物::

    lightgcn_rec/mid/seed_<seed>/best_lightgcn_minibatch.pt

用法::

    python lightgcn_rec/test_lightgcn.py          # 评测 SEEDS: 12 / 22 / 32 / 42 / 52
    python lightgcn_rec/test_lightgcn.py 12 22    # 只评测指定 seed

输出::

    每个 seed 的指标（HR / NDCG / MRR、序列口径指标、precision / recall / f1）
    以及跨 seed 的 mean / std；同时写入 lightgcn_rec/mid/test_result.json
"""

import collections
import json
import math
import os
import pickle as pkl
import sys

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lightgcn_rec.lightgcn import lightGCN  # noqa: F401  # torch.save 的是整个模型，反序列化需要这个类
from utils import L_for_lightGCN, get_scores, test_MRR, test_HR, test_ndcg_100_k

# ------------------------------------------------------------------ 配置
DATA_DIR = os.path.join(PROJECT_ROOT, 'data/MOOCCube/mid_data/need_data') + os.sep
RUN_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mid')

SEEDS = [12]
NUM_NEG = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data():
    adj = np.load(DATA_DIR + 'matrix/adj.npy')
    adj = torch.from_numpy(adj).to(torch.int32)
    ua = torch.from_numpy(np.load(DATA_DIR + 'matrix/ua.npy')).cpu()
    with open(DATA_DIR + 'negatives/test_negative.p', 'rb') as f:
        test_negative = pkl.load(f)
    return adj, ua, test_negative


def get_non_click_matrix(rate, negative, is_score, is_val):
    new_matrix = []
    truth_labels = []
    index = 0
    for u in negative:
        negs = negative[u]
        num_neg = len(negs)
        new_matrix.append(rate[u].view(1, -1).repeat(num_neg, 1))
        truth_labels.extend([neg[-1] for neg in negs])
        index += 1
    new_matrix = torch.cat(new_matrix, dim=0)

    if is_score:
        return new_matrix
    new_matrix = 1 - new_matrix
    new_matrix[range(new_matrix.size(0)), truth_labels] = 1
    return new_matrix, truth_labels


def calculate(score, adj, test_negative, is_val=True):
    topk = 20

    adj, truth_labels = get_non_click_matrix(adj, test_negative, is_score=False, is_val=is_val)
    score = get_non_click_matrix(score, test_negative, is_score=True, is_val=is_val)
    topk_items = []
    batch = 1024
    for index in tqdm(range(0, score.shape[0], batch)):
        score_new = score[index:index + batch]
        non_adj_new = adj[index:index + batch]
        score_new = score_new * non_adj_new
        score_new = np.where(score_new == 0, -math.inf, score_new)
        sorted_items = np.argsort(-score_new, axis=1)
        topk_items_sub = sorted_items[:, :topk].tolist()
        topk_items.extend(topk_items_sub)

    mrr_list = []
    hr_1_list = []
    hr_5_list = []
    hr_10_list = []
    hr_20_list = []
    ndcg_1_list = []
    ndcg_5_list = []
    ndcg_10_list = []
    ndcg_20_list = []

    def calculate_dcg(relevance_scores, k):
        dcg = 0.0
        for i in range(k):
            if i < len(relevance_scores):
                dcg += relevance_scores[i] / math.log2(i + 2)
            else:
                break
        return dcg

    def calculate_idcg(relevance_scores, k):
        sorted_scores = sorted(relevance_scores, reverse=True)
        return calculate_dcg(sorted_scores[:k], k)

    def calculate_ndcg(relevance_scores, k):
        dcg_k = calculate_dcg(relevance_scores, k)
        idcg_k = calculate_idcg(relevance_scores, k)
        if idcg_k == 0:
            return 0.0
        return dcg_k / idcg_k

    for i, ground_truth in enumerate(truth_labels):
        topk_concepts = topk_items[i]
        result = [1 if ground_truth == concept else 0 for concept in topk_concepts]
        try:
            target_rank = result.index(1) + 1
        except ValueError:
            mrr_list.append(0)
            hr_1_list.append(0)
            hr_5_list.append(0)
            hr_10_list.append(0)
            hr_20_list.append(0)
            ndcg_1_list.append(0)
            ndcg_5_list.append(0)
            ndcg_10_list.append(0)
            ndcg_20_list.append(0)
            continue
        mrr_list.append(1.0 / target_rank)
        hr_1_list.append(sum(result[:1]))
        hr_5_list.append(sum(result[:5]))
        hr_10_list.append(sum(result[:10]))
        hr_20_list.append(sum(result[:20]))
        ndcg_20_list.append(calculate_ndcg(result, 20))
        ndcg_10_list.append(calculate_ndcg(result, 10))
        ndcg_5_list.append(calculate_ndcg(result, 5))
        ndcg_1_list.append(calculate_ndcg(result, 1))

    mrr = sum(mrr_list) / len(mrr_list)
    hr_1 = sum(hr_1_list) / len(hr_1_list)
    hr_5 = sum(hr_5_list) / len(hr_5_list)
    hr_10 = sum(hr_10_list) / len(hr_10_list)
    hr_20 = sum(hr_20_list) / len(hr_20_list)
    ndcg_1 = sum(ndcg_1_list) / len(ndcg_1_list)
    ndcg_5 = sum(ndcg_5_list) / len(ndcg_5_list)
    ndcg_10 = sum(ndcg_10_list) / len(ndcg_10_list)
    ndcg_20 = sum(ndcg_20_list) / len(ndcg_20_list)
    print(f'mrr: {mrr}, hr@1: {hr_1}, hr@5: {hr_5}, hr@10: {hr_10}, hr@20: {hr_20}, '
          f'ndcg@1: {ndcg_1}, ndcg@5: {ndcg_5}, ndcg@10: {ndcg_10}, ndcg@20: {ndcg_20}')
    return {'mrr': mrr, 'hr@1': hr_1, 'hr@5': hr_5, 'hr@10': hr_10, 'hr@20': hr_20,
            'ndcg@1': ndcg_1, 'ndcg@5': ndcg_5, 'ndcg@10': ndcg_10, 'ndcg@20': ndcg_20}


def precision_k(sorted_items, test_user_dict, num_u, k):
    precisions = []
    topk_items = sorted_items[:, :k].tolist()
    for i in range(num_u):
        pred_items = set(topk_items[i])
        true_items = test_user_dict[i]
        num_hit = len(pred_items & true_items)
        precisions.append(num_hit / k)
    return sum(precisions) / len(precisions)


def recall_k(sorted_items, test_user_dict, num_u, k):
    recalls = []
    topk_items = sorted_items[:, :k].tolist()
    for i in range(num_u):
        pred_items = set(topk_items[i])
        true_items = test_user_dict[i]
        num_hit = len(pred_items & true_items)
        if len(true_items) == 0:
            recalls.append(0)
        else:
            recalls.append(num_hit / len(true_items))
    return sum(recalls) / len(recalls)


def f1score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def evaluate(seed, adj, ua, test_negative, H_train):
    ckpt = os.path.join(RUN_ROOT, f'seed_{seed}', 'best_lightgcn_minibatch.pt')
    if not os.path.exists(ckpt):
        print(f'[skip] seed {seed}: 没有 {ckpt}')
        return None
    print(f'===== seed {seed} =====')
    model = torch.load(ckpt, weights_only=False, map_location=device)
    model.eval()
    num_u = adj.size(0)

    metrics = {}
    with torch.no_grad():
        U_test, I_test = model(H_train)
        score = U_test.mm(I_test.T)

    # 负例采样口径；test_negative 每用户长度不统一时该口径不适用
    if all(len(test_negative[u]) == NUM_NEG for u in test_negative):
        neg_score = get_scores(test_negative, score.to('cpu'), NUM_NEG)
        for k in (1, 5, 10, 20):
            metrics[f'hr@{k}'] = test_HR(neg_score, k=k, dim=NUM_NEG)
            print(f'hr@{k}: ', metrics[f'hr@{k}'])
        for k in (1, 5, 10, 20):
            metrics[f'ndcg@{k}'] = test_ndcg_100_k(neg_score, k=k, dim=NUM_NEG)
            print(f'ndcg@{k}: ', metrics[f'ndcg@{k}'])
        metrics['mrr'] = test_MRR(neg_score, NUM_NEG)
        print('mrr: ', metrics['mrr'])
    else:
        print(f'[skip] 负例采样口径：test_negative 每用户长度不统一（期望 {NUM_NEG}），该口径不适用')

    # 序列口径（对齐 ua，过滤已交互）
    score = score.cpu()
    score = torch.nn.Sigmoid()(score)
    seq = calculate(score, ua, test_negative, False)
    metrics.update({f'seq_{k}': v for k, v in seq.items()})

    # precision / recall
    test_ua = ua - adj
    test_user_dict = collections.defaultdict(set)
    for x in torch.nonzero(test_ua).tolist():
        test_user_dict[x[0]].add(x[1])
    pred_score = score.numpy() * (1 - adj.numpy())
    print(np.min(pred_score), np.max(pred_score))
    sorted_items = np.argsort(-pred_score, axis=1)

    precision5 = precision_k(sorted_items, test_user_dict, num_u, 5)
    recall5 = recall_k(sorted_items, test_user_dict, num_u, 5)
    f1_5 = f1score(precision5, recall5)
    precision10 = precision_k(sorted_items, test_user_dict, num_u, 10)
    recall10 = recall_k(sorted_items, test_user_dict, num_u, 10)
    f1_10 = f1score(precision10, recall10)
    precision20 = precision_k(sorted_items, test_user_dict, num_u, 20)
    recall20 = recall_k(sorted_items, test_user_dict, num_u, 20)
    f1_20 = f1score(precision20, recall20)

    print('precision@5: ', precision5)
    print('recall@5: ', recall5)
    print('f1@5: ', f1_5)
    print('precision@10: ', precision10)
    print('recall@10: ', recall10)
    print('f1@10: ', f1_10)
    print('precision@20: ', precision20)
    print('recall@20: ', recall20)
    print('f1@20: ', f1_20)

    metrics.update({'precision@5': precision5, 'recall@5': recall5, 'f1@5': f1_5,
                    'precision@10': precision10, 'recall@10': recall10, 'f1@10': f1_10,
                    'precision@20': precision20, 'recall@20': recall20, 'f1@20': f1_20})
    return metrics


def summary(results):
    keys = list(next(iter(results.values())).keys())
    out = {}
    for key in keys:
        values = np.array([m[key] for m in results.values()], dtype=np.float64)
        out[key] = {'mean': float(values.mean()),
                    'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    'n': int(len(values))}
    return out


def main(seeds=None):
    seeds = seeds or SEEDS
    adj, ua, test_negative = load_data()
    H_train = L_for_lightGCN(adj).to(device)

    results = {}
    for seed in seeds:
        metrics = evaluate(seed, adj, ua, test_negative, H_train)
        if metrics:
            results[seed] = metrics
    if not results:
        print('没有可评测的模型')
        return

    stats = summary(results)
    print(f'\n===== mean / std over {len(results)} seeds =====')
    print(f'{"metric":<14s}{"mean":>10s}{"std":>10s}')
    for key, s in stats.items():
        print(f'{key:<14s}{s["mean"]:>10.5f}{s["std"]:>10.5f}')

    out = os.path.join(RUN_ROOT, 'test_result.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'seeds': list(results), 'per_seed': results, 'summary': stats},
                  f, ensure_ascii=False, indent=2)
    print(f'\nsaved: {out}')


if __name__ == '__main__':
    main([int(x) for x in sys.argv[1:]])
