"""stage-1 的数据层：复用 sasrec_old 的口径，从原始交互表构造 train / val / test。
"""

import hashlib
import json
import os
import pickle as pkl
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
for _p in (PROJECT_ROOT, CURRENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sasrec_old.config import Config as SasrecConfig
from sasrec_old.utils import (draw_negatives, flatten_concepts, get_concept_seq,
                              pad_concepts)


def windows_cache_path(cf_cfg: SasrecConfig) -> str:
    """与 sasrec_old 完全相同的缓存文件名，命中就直接复用它的缓存。"""
    need = os.path.join(PROJECT_ROOT, cf_cfg.dataset_dir, 'need_data') + os.sep
    key = {
        'v': 1,
        'dataset_dir': cf_cfg.dataset_dir,
        'train_date_time': cf_cfg.train_date_time,
        'test_date_start': cf_cfg.test_date_start,
        'test_date_end': cf_cfg.test_date_end,
        'max_len': cf_cfg.max_len,
        'window_stride': cf_cfg.window_stride,
        'vk_mtime': os.path.getmtime(need + 'relations/vk.p'),
        'df_mtime': os.path.getmtime(
            os.path.join(PROJECT_ROOT, cf_cfg.dataset_dir, 'filtered_second_df2.pkl')),
    }
    cache_id = hashlib.md5(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return os.path.join(cf_cfg.prepare_data_dir(PROJECT_ROOT), f'concept_windows_{cache_id}.pkl')


def build_windows(cf_cfg: SasrecConfig) -> dict:
    """复刻 sasrec_old 的窗口构造（只在缓存缺失或不兼容时才会走到这里）。"""
    import pandas as pd

    need = os.path.join(PROJECT_ROOT, cf_cfg.dataset_dir, 'need_data')
    vk_dict = pkl.load(open(os.path.join(need, 'relations/vk.p'), 'rb'))

    columns = ['index', 'id']
    user_df = pd.read_csv(os.path.join(need, 'entities/users.csv'), header=None)
    user_df.columns = columns
    user2index = dict(zip(user_df['id'], user_df['index']))
    video_df = pd.read_csv(os.path.join(need, 'entities/videos.csv'), header=None)
    video_df.columns = columns
    video2index = dict(zip(video_df['id'], video_df['index']))

    df = pkl.load(open(os.path.join(PROJECT_ROOT, cf_cfg.dataset_dir,
                                    'filtered_second_df2.pkl'), 'rb'))
    df['id'] = df['id'].apply(lambda x: user2index[x])
    df['vid'] = df['vid'].apply(lambda x: video2index[x])
    df = df.sort_values(by=['id', 'local_start_time'])
    train_val_df = df[(df['local_start_time'] < cf_cfg.test_date_start) &
                      (df['local_start_time'] >= cf_cfg.train_date_time)]
    test_df = df[(df['local_start_time'] < cf_cfg.test_date_end) &
                 (df['local_start_time'] >= cf_cfg.test_date_start)]

    # concept id 整体 +1，0 留给 padding
    vk_dict = {vid + 1: [c + 1 for c in concepts] for vid, concepts in vk_dict.items()}

    train_ua, train_uk, train_next, train_val_uv = {}, {}, {}, {}
    max_len = cf_cfg.max_len
    for uid in train_val_df['id'].unique():
        udf = train_val_df[train_val_df['id'] == uid]
        train_val_uv[uid] = [vid + 1 for vid in udf['vid'].values.tolist()]
        # 训练历史 = val video 之前的所有 video，平铺成 concept 序列
        user_concepts = flatten_concepts(train_val_uv[uid][:-2], vk_dict)
        user_uk = list(set(user_concepts))
        val_next = vk_dict[train_val_uv[uid][-2]][0]
        stride = cf_cfg.window_stride or max_len
        t = len(user_concepts) - max_len
        starts = [0] if t <= 0 else list(range(t, 0, -stride)) + [0]
        for s in reversed(starts):
            window = user_concepts[:s + max_len]
            key = str(uid) + '|' + str(s)
            train_ua[key] = pad_concepts(window, max_len)
            train_uk[key] = user_uk
            train_next[key] = (user_concepts[s + max_len]
                               if s + max_len < len(user_concepts) else val_next)

    val_seq_dict = {uid: get_concept_seq(train_val_uv[uid][:-1], vk_dict, max_len)[0]
                    for uid in train_val_uv}
    test_seq_dict = {uid: get_concept_seq(train_val_uv[uid], vk_dict, max_len)[0]
                     for uid in train_val_uv}

    test_uv = {}
    for uid in test_df['id'].unique():
        udf = test_df[test_df['id'] == uid].sort_values('local_start_time')
        test_uv[uid] = [udf['vid'].values.tolist()[0] + 1]

    return {'train_val_uv': train_val_uv, 'train_ua': train_ua, 'train_uk': train_uk,
            'train_next': train_next, 'val_seq_dict': val_seq_dict,
            'test_seq_dict': test_seq_dict, 'test_uv': test_uv}


def load_windows(cf_cfg: SasrecConfig) -> dict:
    """优先读 sasrec_old 的缓存；numpy 大版本不兼容时重建并另存，不覆盖原缓存。"""
    path = windows_cache_path(cf_cfg)
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                cached = pkl.load(f)
            print(f'windows cache: {path}')
            return cached
        except (ModuleNotFoundError, ImportError, AttributeError) as e:
            print(f'[warn] 缓存与当前 numpy 不兼容（{type(e).__name__}），改为重建：{path}')

    print('building concept windows ...')
    cached = build_windows(cf_cfg)
    out = path.replace('.pkl', f'_np{np.__version__.split(".")[0]}.pkl')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'wb') as f:
        pkl.dump(cached, f, protocol=pkl.HIGHEST_PROTOCOL)
    print(f'windows cache saved: {out}')
    return cached


def get_log_seqs(seq_dict: dict, num_user: int, max_len: int) -> np.ndarray:
    """按 uid 0..num_user-1 拼成 (num_user, max_len)，缺失用户给全 0。"""
    seqs = np.zeros([num_user, max_len], dtype=np.int32)
    for uid in range(num_user):
        seq = seq_dict.get(uid)
        if seq is not None:
            seqs[uid] = seq
    return seqs


class AlignTrainDataset(Dataset):
    """一条样本 = 一个 CF 训练窗口：(seq, 窗口外的下一个 concept, 1 个未交互负例)。

    与 `ConceptNextDataset` 的差别只在「每个窗口取几个位置」：CF 要逐位置算 BPR，
    这里按论文 4.1 只取末位置的 user_feat 和它对应的 target。负例池与排除规则保持一致。
    """

    def __init__(self, windows: dict, num_concept: int):
        self.num_concept = num_concept
        self.ua = windows['train_ua']
        self.next = windows['train_next']
        self.keys = list(self.ua)
        self.uid = {key: int(key.split('|')[0]) for key in self.keys}

        # 同一用户的所有窗口共享同一份历史 list，按 uid 去重成 set
        user_uk = {}
        for key in self.keys:
            uid = self.uid[key]
            if uid not in user_uk:
                user_uk[uid] = set(windows['train_uk'][key])
        self.user_uk = user_uk
        # 末窗口的 next（val video 首个 concept）可能不在历史里，抽负例时要额外排除
        self.extra = {key: (v if v not in user_uk[self.uid[key]] else 0)
                      for key, v in self.next.items()}
        self.pool_cache = {}

    def candidate_pool(self, uid: int) -> np.ndarray:
        """该用户没交互过的 concept 候选池（1..num_concept 去掉历史），按 uid 缓存。"""
        pool = self.pool_cache.get(uid)
        if pool is None:
            mask = np.ones(self.num_concept + 1, dtype=bool)
            mask[0] = False
            mask[list(self.user_uk[uid])] = False
            pool = np.flatnonzero(mask).astype(np.int32)
            self.pool_cache[uid] = pool
        return pool

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        uid = self.uid[key]
        target = int(self.next[key])
        neg = draw_negatives(self.candidate_pool(uid), 1, self.extra[key])
        return (torch.from_numpy(self.ua[key]),
                torch.tensor(target, dtype=torch.long),
                torch.tensor(int(neg[0]), dtype=torch.long))
