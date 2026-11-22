"""sasrec_old 的公共组件：数据构造、负采样、BPR loss 与 top-k 评估。"""

import math

import numpy as np
import torch
from torch.utils.data import Dataset


def shuffle_padded_seq(seq):
    """只打乱实际序列，左侧 padding 0 保持不动。"""
    valid = seq[seq != 0]
    np.random.shuffle(valid)
    return np.concatenate([np.zeros(len(seq) - len(valid), dtype=seq.dtype), valid])


def BPRLoss(pos_pred, neg_pred):
    loss = torch.mean(-torch.log(torch.sigmoid(pos_pred - neg_pred)), dim=0)
    if math.isinf(loss.item()):
        inf_index = torch.where(torch.isinf(torch.log(torch.sigmoid(pos_pred - neg_pred))))
        delta = (pos_pred - neg_pred).clone()
        delta[inf_index] = 0
        min_value = torch.min(delta)
        delta = torch.where(delta < min_value, min_value, delta)
        loss = torch.mean(-torch.log(torch.sigmoid(delta)), dim=0)
    return loss


def flatten_concepts(videos, vk_dict):
    """把 video 序列按 vk 关系平铺成 concept 序列（不截断）。"""
    concepts = []
    for vid in videos:
        concepts.extend(vk_dict[vid])
    return concepts


def pad_concepts(concepts, max_len):
    """取 concept 序列的最后 max_len 个，左侧 padding 0 到定长（numpy 直接切片赋值）。"""
    concepts = np.asarray(concepts, dtype=np.int32)[-max_len:]
    seq = np.zeros([max_len], dtype=np.int32)
    if concepts.size:
        seq[max_len - concepts.size:] = concepts
    return seq


def get_concept_seq(videos, vk_dict, max_len):
    """把 video 序列按 vk 关系平铺成 concept 序列，取最后 max_len 个，左侧 padding 0。"""
    concepts = flatten_concepts(videos, vk_dict)
    return pad_concepts(concepts, max_len), concepts


def build_next_concepts(seq, next_concept):
    """逐位置的下一个 concept
    """
    nxt = np.zeros_like(seq)
    nxt[:-1] = seq[1:]
    if seq[-1] != 0:
        nxt[-1] = next_concept
    nxt[seq == 0] = 0
    return nxt


def shuffle_padded_seq_and_next(seq, nxt):
    """打乱有效位置，并把同一置换作用到 next 目标上，保持二者逐位置对应。"""
    valid = np.flatnonzero(seq)
    perm = np.random.permutation(valid)
    seq_new = seq.copy()
    nxt_new = nxt.copy()
    seq_new[valid] = seq[perm]
    nxt_new[valid] = nxt[perm]
    return seq_new, nxt_new


def draw_negatives(pool, k, forbidden=0):
    """从候选池中不放回抽 k 个负例（numpy 向量化，替代逐次 Python 拒绝采样）。

    pool: 1D int32 数组，该用户从未交互过的 concept（升序，长度约 1.2w）
    forbidden: 该窗口还需额外排除的 target（末窗口的 val_next），0 表示无
    """
    capacity = pool.size - (1 if forbidden else 0)
    if k > capacity:
        k = capacity
    if k <= 0:
        return np.zeros([0], dtype=np.int32)
    # k ~ 100、池 ~ 1.2w，有放回抽样的重复期望 < 1，补抽 1~2 轮即可
    picked = np.unique(pool[np.random.randint(0, pool.size, size=k)])
    if forbidden:
        picked = picked[picked != forbidden]
    while picked.size < k:
        extra = pool[np.random.randint(0, pool.size, size=k - picked.size + 4)]
        if forbidden:
            extra = extra[extra != forbidden]
        picked = np.concatenate([picked, np.setdiff1d(extra, picked)])
    return picked[:k].astype(np.int32)


class ConceptNextDataset(Dataset):
    """每条样本：历史 concept 序列 + 逐位置的下一个 concept。
    """

    def __init__(self, num_k, ua_dict, next_dict, uk_dict, shuffle_seq=False):
        self.num_k = num_k  # concept 总数（不含 padding）
        self.ua_dict = ua_dict
        self.uk_dict = uk_dict
        self.next_dict = next_dict
        self.shuffle_seq = shuffle_seq
        # 每个用户的历史 concept 集合：同一用户的窗口共享同一个 list，按 uid 去重
        self.user_uk_set = {}
        for key, uk in uk_dict.items():
            uid = int(key.split('|')[0])
            if uid not in self.user_uk_set:
                self.user_uk_set[uid] = set(uk)
        # 每个窗口除历史外还需排除的 target：只有末窗口的 next（val video 首个 concept）可能不在历史里
        self.extra = {}
        for key, v in next_dict.items():
            self.extra[key] = v if v not in self.user_uk_set[int(key.split('|')[0])] else 0
        self.pool_cache = {}  # uid -> 候选 concept 池，首次访问时构建
        self.dataset = self.get_sequences(ua_dict, next_dict)

    def get_sequences(self, ua_dict, next_dict):
        dataset = []
        for key in ua_dict:
            seq = ua_dict[key]
            nxt = build_next_concepts(seq, next_dict[key])  # 逐位置的下一个 concept
            dataset.append([key, int(key.split('|')[0]), seq, nxt])
        return dataset

    def candidate_pool(self, uid):
        """该用户没交互过的 concept 候选池（1..num_k 去掉历史），按 uid 缓存复用。"""
        pool = self.pool_cache.get(uid)
        if pool is None:
            mask = np.ones(self.num_k + 1, dtype=bool)
            mask[0] = False  # 0 是 padding
            mask[list(self.user_uk_set[uid])] = False
            pool = np.flatnonzero(mask).astype(np.int32)
            self.pool_cache[uid] = pool
        return pool

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        key, uid, seq, nxt = self.dataset[idx]

        if self.shuffle_seq:
            seq, nxt = shuffle_padded_seq_and_next(seq, nxt)

        # 每个有效位置一个目标 + 一个负例
        nxt_neg = np.zeros_like(nxt)
        pos = np.flatnonzero(nxt)
        if pos.size:
            nxt_neg[pos] = draw_negatives(self.candidate_pool(uid), pos.size, self.extra[key])

        return (torch.from_numpy(seq), torch.from_numpy(nxt), torch.from_numpy(nxt_neg))


def calculate(score, adj, negative, topk=20, return_items=False):
    """1-adj 掩码后的 per-query top-k 评估：mrr / hr@1,5,10,20 / ndcg@1,5,10,20。

    不做 (query × concept) 的矩阵展开：每个用户只对 score 排一次序，然后
    用「比 truth 分数高的未点击项个数」直接算 rank，复杂度从 O(Q*C) 降到 O(U*C log C)。
    """
    score = score.numpy() if torch.is_tensor(score) else np.asarray(score)
    adj = adj.numpy() if torch.is_tensor(adj) else np.asarray(adj)
    score = score.astype(np.float32, copy=False)
    adj = adj.astype(np.float32, copy=False)

    users = list(negative.keys())
    truth_labels = [neg[-1] for u in users for neg in negative[u]]
    truth = np.asarray(truth_labels, dtype=np.int64)

    rank = np.empty(len(truth), dtype=np.int64)
    topk_items = [] if return_items else None
    pos = 0
    for u in users:
        q = len(negative[u])
        srow = score[u]
        arow = adj[u] == 0  # 未点击的候选项
        ts = truth[pos:pos + q]
        # 比 truth 分数高的未点击项个数 + 1 = truth 在候选里的排名
        rank[pos:pos + q] = 1 + np.count_nonzero(
            arow[None, :] & (srow[None, :] > srow[ts][:, None]), axis=1)
        if return_items:
            # 只需前 topk 个未点击项：先 argpartition 截掉大部分，再对少量候选排序
            k = min(topk + int(np.count_nonzero(~arow)), srow.size - 2)
            cand = np.argpartition(-srow, k)[:k + 1]
            sel = cand[arow[cand]][:topk + 1]
            base = sel[np.argsort(-srow[sel])].tolist()
            for r, t in zip(rank[pos:pos + q], ts):
                if (not arow[t]) and r <= topk:
                    items = base[:topk]
                    items.insert(int(r) - 1, int(t))
                    topk_items.append(items[:topk])
                else:
                    topk_items.append(base[:topk])
        pos += q

    hit = rank <= topk
    mrr = float(np.mean(np.where(hit, 1.0 / rank, 0.0)))
    metrics = {'mrr': mrr}
    for k in (1, 5, 10, 20):
        in_k = rank <= k
        metrics[f'hr@{k}'] = float(np.mean(in_k))
        metrics[f'ndcg@{k}'] = float(np.mean(np.where(in_k, 1.0 / np.log2(rank + 1), 0.0)))

    print(f'mrr: {mrr}, hr@1: {metrics["hr@1"]}, hr@5: {metrics["hr@5"]}, '
          f'hr@10: {metrics["hr@10"]}, hr@20: {metrics["hr@20"]}, '
          f'ndcg@1: {metrics["ndcg@1"]}, ndcg@5: {metrics["ndcg@5"]}, '
          f'ndcg@10: {metrics["ndcg@10"]}, ndcg@20: {metrics["ndcg@20"]}')
    if return_items:
        return metrics, topk_items, truth_labels
    return metrics
