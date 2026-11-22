"""LightGCN 训练（mid_data）：逐 epoch 验证，按验证 MRR 保存各 seed 的最优模型。

用法::

    python lightgcn_rec/train_lightgcn.py          # 依次跑 SEEDS: 12 / 22 / 32 / 42 / 52
    python lightgcn_rec/train_lightgcn.py 12 22    # 只跑指定 seed

产物::

    lightgcn_rec/mid/seed_<seed>/best_lightgcn_minibatch.pt
"""

import collections
import math
import os
import pickle as pkl
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lightgcn_rec.lightgcn import lightGCN
from utils import L_for_lightGCN, UIDataset, BPRLoss, l2Loss

# ------------------------------------------------------------------ 配置
DATA_DIR = os.path.join(PROJECT_ROOT, 'data/MOOCCube/mid_data/need_data') + os.sep
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mid')

SEEDS = [42]

EMB_DIM = 256
NUM_LAYERS = 2
EPOCHS = 150
LR = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.1
BATCH_SIZE = 1024

# 设置训练设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def load_data():
    adj = np.load(DATA_DIR + 'matrix/adj.npy')
    rate = np.load(DATA_DIR + 'matrix/rate.npy')
    adj = torch.from_numpy(adj).to(torch.int32)
    rate = torch.from_numpy(rate).to(torch.int32)
    with open(DATA_DIR + 'negatives/negative.p', 'rb') as f:
        negative = pkl.load(f)
    print('negative.len: ', len(negative))
    return adj, rate, negative


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


def build_dataloaders(adj, rate):
    num_u, num_k = adj.size(0), adj.size(1)
    print('user: ', num_u)
    print('concept: ', num_k)

    # 创建dataloader
    user_item_list = torch.nonzero(rate).tolist()
    uk_dict = collections.defaultdict(list)
    for x in user_item_list:
        uk_dict[x[0]].append(x[1])
    dataset = UIDataset(num_u, num_k, user_item_list, uk_dict)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    ua_lastone = adj - rate
    val_list = torch.nonzero(ua_lastone).tolist()
    adj_list = torch.nonzero(adj).tolist()
    adj_dict = collections.defaultdict(list)
    for x in adj_list:
        adj_dict[x[0]].append(x[1])
    val_dataset = UIDataset(num_u, num_k, val_list, adj_dict)
    val_dataloader = DataLoader(val_dataset, batch_size=num_u, shuffle=False, num_workers=0)

    print('load data done')
    return dataloader, val_dataloader


def train(model, dataloader, optimizer, loss_fn, weight_decay, H_train, device):
    model.train()
    loss_mean = 0
    for user, pos, neg in dataloader:
        user, pos, neg = user.to(device), pos.to(device), neg.to(device)
        optimizer.zero_grad()
        U_out, I_out = model(H_train)
        loss = loss_fn(U_out, I_out, user, pos, neg)
        # 添加正则化项
        u_weight = model.Emb_u.weight[user]
        loss += l2Loss(u_weight, weight_decay)
        pos_weight = model.Emb_k.weight[pos]
        loss += l2Loss(pos_weight, weight_decay)
        neg_weight = model.Emb_k.weight[neg]
        loss += l2Loss(neg_weight, weight_decay)

        loss.backward()
        optimizer.step()
        loss_mean += loss.item() * user.size(0)
    loss_mean /= len(dataloader.dataset)
    return loss_mean


def validate(model, dataloader, loss_fn, H_train, device):
    loss_mean = 0
    for user, pos, neg in dataloader:
        user, pos, neg = user.to(device), pos.to(device), neg.to(device)
        U_out, I_out = model(H_train)
        loss = loss_fn(U_out, I_out, user, pos, neg)
        loss_mean += loss.item() * user.size(0)

    loss_mean /= len(dataloader.dataset)
    return loss_mean


def calculate(score, adj, test_negative, is_val=True):
    topk = 10

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
    ndcg_1_list = []
    ndcg_5_list = []
    ndcg_10_list = []

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
            ndcg_1_list.append(0)
            ndcg_5_list.append(0)
            ndcg_10_list.append(0)
            continue
        mrr_list.append(1.0 / target_rank)
        hr_1_list.append(sum(result[:1]))
        hr_5_list.append(sum(result[:5]))
        hr_10_list.append(sum(result[:10]))
        ndcg_10_list.append(calculate_ndcg(result, 10))
        ndcg_5_list.append(calculate_ndcg(result, 5))
        ndcg_1_list.append(calculate_ndcg(result, 1))

    mrr = sum(mrr_list) / len(mrr_list)
    hr_1 = sum(hr_1_list) / len(hr_1_list)
    hr_5 = sum(hr_5_list) / len(hr_5_list)
    hr_10 = sum(hr_10_list) / len(hr_10_list)
    ndcg_1 = sum(ndcg_1_list) / len(ndcg_1_list)
    ndcg_5 = sum(ndcg_5_list) / len(ndcg_5_list)
    ndcg_10 = sum(ndcg_10_list) / len(ndcg_10_list)
    print(f'mrr: {mrr}, hr@1: {hr_1}, hr@5: {hr_5}, hr@10: {hr_10}, '
          f'ndcg@1: {ndcg_1}, ndcg@5: {ndcg_5}, ndcg@10: {ndcg_10}')
    return {'mrr': mrr, 'hr@1': hr_1, 'hr@5': hr_5, 'hr@10': hr_10,
            'ndcg@1': ndcg_1, 'ndcg@5': ndcg_5, 'ndcg@10': ndcg_10}


def run_seed(seed, adj, negative, dataloader, val_dataloader, num_u, num_k, H_train):
    print(f'===== seed {seed} =====')
    set_seed(seed)

    # 定义模型
    model = lightGCN(num_u, num_k, EMB_DIM, NUM_LAYERS, DROPOUT).to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # loss
    loss_fn = BPRLoss

    out_dir = os.path.join(OUT_ROOT, f'seed_{seed}')
    os.makedirs(out_dir, exist_ok=True)

    # train model
    best_model_mrr = 0
    t_total = time.time()
    for epoch in range(0, EPOCHS):
        loss = train(model, dataloader, optimizer, loss_fn, WEIGHT_DECAY, H_train, device)
        with torch.no_grad():
            model.eval()
            val_loss = validate(model, val_dataloader, loss_fn, H_train, device)

            U_out, K_out = model(H_train)
            score = U_out.mm(K_out.t())
            val_score = torch.nn.Sigmoid()(score).cpu()
            print('epoch: ', epoch, ' loss: ', loss, ' val_loss: ', val_loss)
            val_mrr = calculate(val_score, adj, negative)['mrr']

        if val_mrr > best_model_mrr:
            best_model_mrr = val_mrr
            torch.save(model, os.path.join(out_dir, 'best_lightgcn_minibatch.pt'))

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print(f'seed {seed} best val mrr: {best_model_mrr}')
    return best_model_mrr


def main(seeds=None):
    adj, rate, negative = load_data()
    num_u, num_k = adj.size(0), adj.size(1)
    dataloader, val_dataloader = build_dataloaders(adj, rate)

    # 求train_uk的拉普拉斯矩阵
    H_train = L_for_lightGCN(adj)
    H_train = H_train.to(device)

    for seed in seeds or SEEDS:
        run_seed(seed, adj, negative, dataloader, val_dataloader, num_u, num_k, H_train)


if __name__ == '__main__':
    main([int(x) for x in sys.argv[1:]])
