import random

import numpy as np
import torch
from torch.utils.data import Dataset
import math


def shuffle_padded_seq(seq):
    # 只打乱实际序列，左侧 padding 0 保持不动
    valid = seq[seq != 0]
    np.random.shuffle(valid)
    return np.concatenate([np.zeros(len(seq) - len(valid), dtype=seq.dtype), valid])


class SequenceAugDataset(Dataset):
    def __init__(self, num_i, num_k, ua_dict, label_dict, uk_dict, vk_dict, max_len, shuffle_seq=False):
        self.num_i = num_i  # video
        self.num_k = num_k  # concept
        self.shuffle_seq = shuffle_seq
        self.ua_dict = ua_dict
        self.uk_dict = uk_dict
        self.vk_dict = vk_dict
        self.max_concepts = max([len(value) for value in vk_dict.values()])
        self.label_dict = label_dict
        self.max_len = max_len
        self.dataset = self.get_sequence(ua_dict)

    def get_sequence(self, ua_dict):
        def one_sample(ts, max_len, nxt):
            idx = 0
            seq = np.zeros([max_len], dtype=np.int32)
            pos = np.zeros([max_len], dtype=np.int32)
            for t in reversed(ts):
                seq[self.max_len-idx-1] = t
                pos[self.max_len-idx-1] = nxt
                nxt = t
                idx += 1
                if idx == max_len:
                    break
            return seq, pos

        dataset = []
        for user in ua_dict:
            ts = ua_dict[user]
            nxt = self.label_dict[user][0]
            seq, pos = one_sample(ts, self.max_len, nxt)
            dataset.append([user, seq, pos])
        return dataset

    def __len__(self):
        return len(self.dataset)

    def get_negs(self, user):
        excluded = set(self.ua_dict[user]) | set(self.label_dict[user])
        return np.array(self.sample_without_replacement(self.num_i, excluded, self.max_len), dtype=np.int32)

    def get_neg_concepts(self, user, pos_concepts, max_len):
        excluded = set(self.uk_dict[user]) | set(pos_concepts)
        return self.sample_without_replacement(self.num_k, excluded, max_len)

    @staticmethod
    def sample_without_replacement(num_total, excluded, k):
        # 候选集远大于 k 时用拒绝采样，避免每次采样都构造上万元素的 set 差集
        if num_total - len(excluded) < 10 * k:
            pool = [i for i in range(1, num_total + 1) if i not in excluded]
            return random.sample(pool, k)
        sampled = set()
        while len(sampled) < k:
            i = random.randint(1, num_total)
            if i not in excluded:
                sampled.add(i)
        return list(sampled)

    def __getitem__(self, idx):
        user, seq, pos = self.dataset[idx]

        if self.shuffle_seq:
            seq = shuffle_padded_seq(seq)

        pos_concepts = np.array(self.vk_dict[pos[-1]])
        neg_concepts = np.array(self.get_neg_concepts(user, pos_concepts, len(pos_concepts)))

        neg = self.get_negs(user)
        neg = np.where(pos==0, 0, neg)
        seq, pos, neg = torch.from_numpy(seq), torch.from_numpy(pos), torch.from_numpy(neg)
        #################
        if len(pos_concepts) < self.max_concepts:
            pos_concepts = np.concatenate([pos_concepts, np.array([0]*(self.max_concepts-len(pos_concepts)))])
            neg_concepts = np.concatenate([neg_concepts, np.array([0]*(self.max_concepts-len(neg_concepts)))])
        pos_concepts = torch.from_numpy(pos_concepts)
        neg_concepts = torch.from_numpy(neg_concepts)
        return user, seq, pos, neg, pos_concepts, neg_concepts

def BPRLoss(pos_pred, neg_pred):
    loss = torch.mean(-torch.log(torch.sigmoid(pos_pred - neg_pred)), dim=0)
    if math.isinf(loss.item()):
        inf_index = torch.where(torch.isinf(torch.log(torch.sigmoid(pos_pred - neg_pred))))
        delta = pos_pred - neg_pred
        delta_temp = delta.clone()
        delta_temp[inf_index] = 0
        min = torch.min(delta_temp)
        delta = torch.where(delta < min, min, delta)
        loss = torch.mean(-torch.log(torch.sigmoid(delta)), dim=0)
    return loss

def get_scores(negative, y_pred, dim=100):
    num_u = y_pred.size(0)
    neg_score = []
    for u in range(num_u):
        negative_u = negative[u]
        for neg in negative_u:
            neg_score_u = y_pred[u].gather(0, torch.tensor(neg))
            neg_score.append(neg_score_u)
    # 对样例匹配得分
    neg_score = torch.cat(neg_score, dim=0).view(-1, dim)
    return neg_score


def get_val_scores(negative, y_pred, dim=100):
    num_u = y_pred.size(0)
    neg_score = []
    for u in range(num_u):
        negative_u = negative[u]
        neg_score_u = y_pred[u].gather(0, torch.tensor(negative_u))
        neg_score.append(neg_score_u)
    # 对样例匹配得分
    neg_score = torch.cat(neg_score, dim=0).view(-1, dim)
    return neg_score


# 实现mrr
def MRR(neg_score, dim=100):
    # 求前k个的下标
    topk = torch.topk(neg_score, k=dim, dim=1).indices
    # 找到下标为99的正例所在的位置
    topk_index = torch.where(topk == dim-1)[1]
    mrr = torch.mean(1 / (topk_index.float() + 1))
    return mrr.item()


def test_MRR(neg_score, dim=100):
    # 求前k个的下标
    topk = torch.topk(neg_score, k=dim, dim=1).indices
    # 找到下标为99的正例所在的位置
    topk_index = torch.where(topk == dim-1)[1]
    mrr = torch.mean(1 / (topk_index.float() + 1))
    return mrr.item()


# 实现hr@k
def HR(neg_score, k, dim=100):
    # 求前k个的下标
    topk = torch.topk(neg_score, k=k, dim=1).indices
    # 查看topk中是否存在下标为99的正例
    res = torch.where(topk == dim-1, 1, 0)
    hr = res.sum(dim=1).float().mean().item()
    return hr


def test_HR(neg_score, k, dim=100):
    # 求前k个的下标
    topk = torch.topk(neg_score, k=k, dim=1).indices
    # 查看topk中是否存在下标为99的正例
    res = torch.where(topk == dim-1, 1, 0)
    hr = res.sum(dim=1).float().mean().item()
    return hr


# 实现ndcg@k
def _dcg(matrix: torch.Tensor) -> torch.Tensor:
    assert matrix.dim() == 2, "The input must be a 2-D tensor."
    n, k = matrix.shape
    denom = (
        torch.log2(torch.arange(k) + 2).view(1, -1).repeat(n, 1)
    )
    return (matrix / denom).sum(dim=1)


# 对100个样例进行ndcg
def ndcg_100_k(neg_score, k, dim=100):
    # 求前k个的下标
    topk = torch.topk(neg_score, k=k, dim=1).indices
    # 查看topk中是否存在下标为99的正例
    res = torch.where(topk == dim-1, 1, 0)

    # pred序列
    pred_dcg = _dcg(res)
    # ideal序列
    ideal_res = torch.zeros_like(res[0])
    ideal_res[0] = 1
    ideal_res = ideal_res.repeat(res.size(0), 1)
    ideal_dcg = _dcg(ideal_res)

    res_list = (pred_dcg / ideal_dcg)
    ndcg = res_list.mean().item()
    return ndcg


def test_ndcg_100_k(neg_score, k, dim=100):
    # 求前k个的下标
    topk = torch.topk(neg_score, k=k, dim=1).indices
    # 查看topk中是否存在下标为99的正例
    res = torch.where(topk == dim-1, 1, 0)

    # pred序列
    pred_dcg = _dcg(res)
    # ideal序列
    ideal_res = torch.zeros_like(res[0])
    ideal_res[0] = 1
    ideal_res = ideal_res.repeat(res.size(0), 1)
    ideal_dcg = _dcg(ideal_res)

    res_list = (pred_dcg / ideal_dcg)
    ndcg = res_list.mean().item()
    return ndcg


# AUC
def AUC(neg_score, dim=100, device='cpu'):
    neg_score = neg_score.to(device)
    # 求前k个的下标
    topk = torch.topk(neg_score, k=dim, dim=1).indices
    # 查看topk中是否存在下标为99的正例
    topk_index = torch.where(topk == dim-1)[1].float()
    ran_auc = torch.randint(0, dim, (len(topk_index), 1)).float().to(device)
    auc = torch.mean(torch.lt(topk_index - ran_auc, 0), dtype=torch.float32)
    del ran_auc
    return auc.item()
