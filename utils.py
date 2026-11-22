import collections
import pickle as pkl
import random

import numpy as np
import torch
from torch.utils.data import Dataset

def get_martix(adj, uc, uv, ck):
    # get meta path
    adj = adj.to(torch.float32)
    uc = uc.to(torch.float32)
    uv = uv.to(torch.float32)
    ck = ck.to(torch.float32)
    uku = torch.mm(adj, adj.T)
    kuk = torch.mm(adj.T, adj)
    ucu = torch.mm(uc, uc.T)
    uvu = torch.mm(uv, uv.T)
    kck = torch.mm(ck.T, ck)

    return uku, kuk, ucu, uvu, kck

def getLrAndLl(H):
    H = H.float()
    diag_v = torch.sum(H, dim=1)
    diag_v_1_2 = torch.where(diag_v > 0, torch.pow(diag_v, -0.5), 0)
    diag_e = torch.sum(H, dim=0)
    diag_e_1 = torch.where(diag_e > 0, torch.pow(diag_e, -1), 0)
    D_v_neg_1_2 = torch.diag(diag_v_1_2).to_sparse_coo()
    D_e_neg_1 = torch.diag(diag_e_1).to_sparse_coo()

    H = H.to_sparse_coo()
    Lr = D_e_neg_1.mm(H.T).mm(D_v_neg_1_2)
    Ll = D_v_neg_1_2.mm(H)
    return Lr, Ll

def Laplacian_matrix(matrix):
    # get Laplacian matrix
    matrix = matrix + torch.eye(matrix.size(0))
    D_v = torch.sum(matrix, dim=1)
    diag_v_1_2 = torch.where(D_v > 0, torch.pow(D_v, -0.5), 0)
    D_v_1_2 = torch.diag(diag_v_1_2).to_sparse_coo()
    matrix = matrix.to_sparse_coo()
    L = D_v_1_2.mm(matrix).mm(D_v_1_2)
    return L

def get_prompt(dataset_name):
    if "Toy" in dataset_name:
        instruction_prompt = "Given a list of toys the user has played before, please recommend a new toy that the user likes to the user."
        history_prompt = "The user has played the following toys before: "
    elif "Amazon_Books" in dataset_name:
        instruction_prompt = "Given a list of books the user has read before, please recommend a new book that the user likes to the user."
        history_prompt = "The user has read the following books before: "
    elif "Clothing" in dataset_name:
        instruction_prompt = "Given a list of clothing the user has worn before, please recommend a new clothing that the user likes to the user."
        history_prompt = "The user has worn the following clothing before: "
    elif "Office" in dataset_name:
        instruction_prompt = "Given a list of office products the user has used before, please recommend a new office product that the user likes to the user."
        history_prompt = "The user has used the following office products before: "

    return instruction_prompt, history_prompt


# 从交互记录变成lightgcn的矩阵 并返回拉普拉斯矩阵
def L_for_lightGCN(B):
    O1 = torch.zeros((B.size(0), B.size(0)))
    hi = torch.cat((O1, B), dim=1)
    O2 = torch.zeros((B.size(1), B.size(1)))
    lo = torch.cat((B.T, O2), dim=1)
    A = torch.cat((hi, lo), dim=0).float()
    # A = A + torch.eye(A.size(0)).float()  # 加上自环
    diag_v = torch.sum(A, dim=1)
    diag_v_1_2 = torch.where(diag_v > 0, torch.pow(diag_v, -0.5), 0)
    D_v_neg_1_2 = torch.diag(diag_v_1_2).to_sparse_coo()
    A = A.to_sparse_coo()
    H = D_v_neg_1_2.mm(A).mm(D_v_neg_1_2)
    return H

def load_self(path):
    # Load data
    adj = np.load(path + 'matrix/adj.npy')
    rate = np.load(path + 'matrix/rate.npy')
    train_uc = np.load(path + 'matrix/train_val_uc.npy')
    train_uv = np.load(path + 'matrix/train_val_uv.npy')
    ck = np.load(path + 'matrix/ck.npy')
    vk = np.load(path + 'matrix/vk.npy')

    # ==================================================================================================================
    adj = torch.from_numpy(adj).to(torch.int32)
    print('adj.shape: ', adj.shape)
    rate = torch.from_numpy(rate).to(torch.int32)
    print('rate.shape: ', rate.shape)
    train_uc = torch.from_numpy(train_uc).to(torch.int32)
    print('uc.shape: ', train_uc.shape)
    print(torch.nonzero((train_uc == 0).all(dim=1)).squeeze())
    train_uv = torch.from_numpy(train_uv).to(torch.int32)
    print('uv.shape: ', train_uv.shape)
    print(torch.nonzero((train_uv == 0).all(dim=1)).squeeze())
    ck = torch.from_numpy(ck).to(torch.int32)
    print('ck.shape: ', ck.shape)
    print(torch.nonzero((ck == 0).all(dim=1)).squeeze())
    vk = torch.from_numpy(vk).to(torch.int32)
    print('vk.shape: ', vk.shape)
    print(torch.nonzero((vk == 0).all(dim=1)).squeeze().shape)
    # zero_videos = torch.nonzero((vk == 0).all(dim=1)).squeeze().numpy().tolist()
    # test_uv = pkl.load(open(path + 'relations/test_uv.p', 'rb'))
    # for user in test_uv:
    #     test_videos = test_uv[user]
    #     print(len(set(zero_videos) & set(test_videos)) / len(test_videos))

    # ==================================================================================================================
    with open(path + 'negatives/negative99.p', 'rb') as f:
        negative = pkl.load(f)
    print('negative.len: ', len(negative))
    with open(path + 'negatives/test_negative99.p', 'rb') as f:
        test_negative = pkl.load(f)
    print('test_negative.len: ', len(test_negative))

    return adj, rate, train_uc, train_uv, ck, vk, negative, test_negative

class UIDataset(Dataset):
    def __init__(self, num_u, num_i, user_item_list, uk_dict, mode='uniform', alpha=1.0):
        self.num_u = num_u
        self.num_i = num_i
        self.data = user_item_list
        self.uk = uk_dict
        self.alpha = alpha
        self.mode = mode
        if mode == 'popularity':
            self.prob = self.count_popularity()

    def count_popularity(self):
        prob = collections.defaultdict(int)
        for ui in self.data:
            prob[ui[1]] += 1
        total = sum(prob.values())
        for item in prob:
            prob[item] = (prob[item] / total) ** self.alpha
        normalize_factor = sum(prob.values())
        for item in prob:
            prob[item] /= normalize_factor
        return prob

    def load_neg(self, user):
        neg_item = random.randrange(self.num_i)
        while neg_item in self.uk[user]:
            neg_item = random.randrange(self.num_i)
        return neg_item

    def load_neg_pop(self, user):
        all_items = list(range(self.num_i))
        pop_scores = [self.prob[item] for item in all_items]
        sample = np.random.choice(all_items, p=pop_scores)
        sample = int(sample)
        while sample in self.uk[user]:
            sample = np.random.choice(all_items, p=pop_scores)
            sample = int(sample)
        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        pos = self.data[idx][1]
        if self.mode == 'popularity':
            neg = self.load_neg_pop(user)
            return user, pos, neg
        neg = self.load_neg(user)
        return user, pos, neg

def l2Loss(emb, l2):
    return 1 / 2 * emb.pow(2).sum(dim=1).mean() * l2


# 实现贝叶斯个性化损失
def BPRLoss(U, I, user, pos, neg):
    emb_u = U[user]
    emb_pos = I[pos]
    emb_neg = I[neg]
    pos_pred = torch.sum(emb_u * emb_pos, dim=1)
    neg_pred = torch.sum(emb_u * emb_neg, dim=1)
    loss = torch.mean(-torch.log(torch.sigmoid(pos_pred - neg_pred)), dim=0)
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