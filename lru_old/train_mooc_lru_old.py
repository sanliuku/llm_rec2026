"""原始 LRU 链路的训练脚本（config 驱动）。

数据 / 窗口 / 损失与 sasrec_old 完全一致（直接复用 sasrec_old/utils.py 的
ConceptNextDataset、BPRLoss 与指标计算），只把模型换成 LRU。
"""

import json
import hashlib
import os
import pickle as pkl
import random
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SASREC_OLD_DIR = os.path.join(PROJECT_ROOT, 'sasrec_old')
sys.path.insert(0, SASREC_OLD_DIR)
sys.path.insert(0, CURRENT_DIR)

from config import Config
from mooc_lru_old import LRUOld
from utils import (BPRLoss, ConceptNextDataset, calculate, flatten_concepts,
                   get_concept_seq, pad_concepts, shuffle_padded_seq)


class _Tee:
    """把 stdout 同时写到终端和 run 目录下的 train.log。"""

    def __init__(self, stream, file):
        self._stream = stream
        self._file = file

    def write(self, msg):
        self._stream.write(msg)
        self._file.write(msg)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def _windows_cache_path(config, path):
    """窗口结果缓存路径：按数据集划分 + 窗口参数 + 原始文件 mtime 生成，放在 prepare_data 下。"""
    key = {
        'v': 1,
        'dataset_dir': config.dataset_dir,
        'train_date_time': config.train_date_time,
        'test_date_start': config.test_date_start,
        'test_date_end': config.test_date_end,
        'max_len': config.max_len,
        'window_stride': config.window_stride,
        'vk_mtime': os.path.getmtime(path + 'relations/vk.p'),
        'df_mtime': os.path.getmtime(os.path.join(PROJECT_ROOT, config.dataset_dir, 'filtered_second_df2.pkl')),
    }
    cache_id = hashlib.md5(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return os.path.join(config.prepare_data_dir(PROJECT_ROOT), f'concept_windows_{cache_id}.pkl')


def load_data(config):
    path = os.path.join(PROJECT_ROOT, config.dataset_dir, 'need_data') + '/'

    train_val_uk_matrix = torch.from_numpy(np.load(path + 'matrix/adj.npy')).float()
    total_uk_matrix = torch.from_numpy(np.load(path + 'matrix/ua.npy'))
    negative = pkl.load(open(path + 'negatives/negative.p', 'rb'))
    test_negative = pkl.load(open(path + 'negatives/test_negative.p', 'rb'))
    vk_dict = pkl.load(open(path + 'relations/vk.p', 'rb'))

    columns = ['index', 'id']
    user_df = pd.read_csv(path + 'entities/users.csv', header=None)
    user_df.columns = columns
    user2index = dict(zip(user_df['id'], user_df['index']))
    video_df = pd.read_csv(path + 'entities/videos.csv', header=None)
    video_df.columns = columns
    video2index = dict(zip(video_df['id'], video_df['index']))

    num_user = len(user_df)
    num_video = len(video_df)
    num_concept = len(pd.read_csv(path + 'entities/concepts.csv', header=None))

    cache_path = _windows_cache_path(config, path)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cached = pkl.load(f)
        print(f'loaded windows cache: {cache_path}')
        train_val_uv = cached['train_val_uv']
        train_ua = cached['train_ua']
        train_uk = cached['train_uk']
        train_next = cached['train_next']
        val_seq_dict = cached['val_seq_dict']
        test_seq_dict = cached['test_seq_dict']
        test_uv = cached['test_uv']
    else:
        filtered_df = pkl.load(open(os.path.join(PROJECT_ROOT, config.dataset_dir, 'filtered_second_df2.pkl'), 'rb'))
        filtered_df['id'] = filtered_df['id'].apply(lambda x: user2index[x])
        filtered_df['vid'] = filtered_df['vid'].apply(lambda x: video2index[x])
        filtered_df = filtered_df.sort_values(by=['id', 'local_start_time'])
        train_val_df = filtered_df[(filtered_df['local_start_time'] < config.test_date_start) &
                                   (filtered_df['local_start_time'] >= config.train_date_time)]
        test_df = filtered_df[(filtered_df['local_start_time'] < config.test_date_end) &
                              (filtered_df['local_start_time'] >= config.test_date_start)]
        print('loading data start....')

        # vk -> concept id 整体 +1，0 留给 padding
        vk_dict = {vid + 1: [concept + 1 for concept in concepts] for vid, concepts in vk_dict.items()}

        train_ua = {}
        train_uk = {}
        train_next = {}
        train_val_uv = {}
        max_len = config.max_len
        for uid in train_val_df['id'].unique():
            udf = train_val_df[train_val_df['id'] == uid]
            train_val_uv[uid] = [vid + 1 for vid in udf['vid'].values.tolist()]
            # 训练历史 = val video 之前的所有 video，直接平铺成 concept 序列
            user_concepts = flatten_concepts(train_val_uv[uid][:-2], vk_dict)
            assert train_val_uv[uid][-2] not in train_val_uv[uid][:-2]
            user_uk = list(set(user_concepts))  # 该用户全部历史 concepts，做负例排除
            val_next = vk_dict[train_val_uv[uid][-2]][0]  # 最后一个窗口的 next 接 val video 首个 concept
            # 从末尾向前按 concept 切窗口：最后一个窗口的 next 一定是 val video 的首个 concept
            stride = config.window_stride or max_len
            t = len(user_concepts) - max_len
            starts = [0] if t <= 0 else list(range(t, 0, -stride)) + [0]
            for s in reversed(starts):
                window = user_concepts[:s + max_len]
                key = str(uid) + '|' + str(s)
                train_ua[key] = pad_concepts(window, max_len)
                train_uk[key] = user_uk
                # 窗口内逐位置右移由 build_next_concepts 完成，这里只补窗口外的那一个 concept
                train_next[key] = (user_concepts[s + max_len]
                                   if s + max_len < len(user_concepts) else val_next)

        val_seq_dict = {uid: get_concept_seq(train_val_uv[uid][:-1], vk_dict, config.max_len)[0]
                        for uid in train_val_uv}
        test_seq_dict = {uid: get_concept_seq(train_val_uv[uid], vk_dict, config.max_len)[0]
                         for uid in train_val_uv}

        test_uv = {}
        for uid in test_df['id'].unique():
            udf = test_df[test_df['id'] == uid].sort_values('local_start_time')
            test_uv[uid] = [udf['vid'].values.tolist()[0] + 1]

        os.makedirs(config.prepare_data_dir(PROJECT_ROOT), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pkl.dump({
                'train_val_uv': train_val_uv,
                'train_ua': train_ua,
                'train_uk': train_uk,
                'train_next': train_next,
                'val_seq_dict': val_seq_dict,
                'test_seq_dict': test_seq_dict,
                'test_uv': test_uv,
            }, f, protocol=pkl.HIGHEST_PROTOCOL)
        print(f'saved windows cache: {cache_path}')

    print(f'train users: {len(train_val_uv)}, train samples: {len(train_ua)}')

    dataset = ConceptNextDataset(num_k=num_concept, ua_dict=train_ua, next_dict=train_next,
                                 uk_dict=train_uk, shuffle_seq=config.shuffle_seq)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    print('load data done')

    return {
        'path': path,
        'train_val_uk_matrix': train_val_uk_matrix,
        'total_uk_matrix': total_uk_matrix,
        'negative': negative,
        'test_negative': test_negative,
        'train_val_uv': train_val_uv,
        'test_uv': test_uv,
        'num_user': num_user,
        'num_video': num_video,
        'num_concept': num_concept,
        'val_seq_dict': val_seq_dict,
        'test_seq_dict': test_seq_dict,
        'dataloader': dataloader,
    }


def get_log_seqs(seq_dict, num_user, max_len, device, shuffle=False):
    log_seqs = []
    for uid in range(num_user):
        seq = seq_dict.get(uid, np.zeros([max_len], dtype=np.int32))
        if shuffle:
            seq = shuffle_padded_seq(seq)
        log_seqs.append(seq.reshape(1, -1))
    return torch.from_numpy(np.concatenate(log_seqs, axis=0)).to(device)


def build_model(config: Config, concept_num: int, device):
    model_config = SimpleNamespace(
        concept_num=concept_num,
        emb_dim=config.emb_dim,
        max_len=config.max_len,
        dropout_rate=config.dropout_rate,
        num_blocks=config.num_blocks,
        lru_r_min=config.lru_r_min,
        lru_r_max=config.lru_r_max,
        device=config.device,
    )
    return LRUOld(model_config).to(device)


def train(model, dataloader, optimizer, device, weight_decay, l2_reg):
    model.train()
    loss_mean = 0
    for seq, next_concepts, next_neg in dataloader:
        seq, next_concepts, next_neg = seq.to(device), next_concepts.to(device), next_neg.to(device)
        optimizer.zero_grad()

        next_pos_logits, next_neg_logits = model(seq, next_concepts, next_neg)

        nindices = torch.where(next_concepts != 0)
        # 逐位置的下一个 concept loss
        loss = BPRLoss(next_pos_logits[nindices], next_neg_logits[nindices])

        # L2 约束：concept 表单独一项，其余参数（LRU / FFN / LayerNorm）合并一项
        loss = loss + weight_decay * torch.norm(model.embedding.concept_emb.weight)
        l2_others = 0.0
        for name, param in model.named_parameters():
            if 'concept_emb.weight' in name:
                continue
            l2_others = l2_others + torch.norm(param)
        loss = loss + l2_reg * l2_others

        loss.backward()
        optimizer.step()
        loss_mean += loss.item() * seq.size(0)
    loss_mean /= len(dataloader.dataset)
    return loss_mean


def evaluate(model, log_seqs, adj, negative):
    score = model.predict_concept(log_seqs)
    score = torch.nn.Sigmoid()(score).cpu().detach()
    return calculate(score[:, 1:], adj, negative)


def main(config: Config):
    seed_everything(config.seed)
    device = torch.device(config.device)
    run_dir = config.create_run_dir(PROJECT_ROOT)
    log_file = open(os.path.join(run_dir, 'train.log'), 'a', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, log_file)

    try:
        print(f'run dir : {run_dir}')
        data = load_data(config)

        model = build_model(config, data['num_concept'], device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        optimizer.zero_grad()

        val_seqs = get_log_seqs(data['val_seq_dict'], data['num_user'], config.max_len, device,
                                shuffle=config.shuffle_seq)
        test_seqs = get_log_seqs(data['test_seq_dict'], data['num_user'], config.max_len, device,
                                 shuffle=config.shuffle_seq)

        max_val = 0.0
        best_test_mrr = 0.0
        init_index = 0
        metrics_path = os.path.join(run_dir, 'metrics.jsonl')
        for epoch in range(config.epochs):
            train_loss = train(model, data['dataloader'], optimizer, device,
                               config.weight_decay, config.l2_reg)
            with torch.no_grad():
                model.eval()
                val_mrr = evaluate(model, val_seqs, data['train_val_uk_matrix'],
                                   data['negative'])['mrr']
                if epoch % config.eval_interval == 0:
                    best_test_mrr = evaluate(model, test_seqs, data['total_uk_matrix'],
                                             data['test_negative'])['mrr']

            if val_mrr > max_val:
                torch.save(model.state_dict(), os.path.join(run_dir, 'best_lru_old.pth'))
                max_val = val_mrr
                init_index = 0
            else:
                init_index += 1

            print(f'epoch: {epoch}, train_loss: {train_loss}, '
                  f'val_mrr: {val_mrr}, mrr: {best_test_mrr}')
            with open(metrics_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_mrr': val_mrr,
                    'test_mrr': best_test_mrr,
                }, ensure_ascii=False) + '\n')

            if init_index > config.early_stop:
                print(f'early stop at epoch {epoch}')
                break

        print('train done')
    finally:
        sys.stdout = original_stdout
        log_file.close()


if __name__ == '__main__':
    config = Config()
    config.dataset_dir = 'data/MOOCCube/small_data'
    config.train_date_time = '2019-01-01 00:00:00'
    config.test_date_start = '2019-05-01 00:00:00'
    config.test_date_end = '2019-06-01 00:00:00'

    seeds = [22, 32, 42, 52]
    # seeds = [12]
    for seed in seeds:
        config.seed = seed
        config.exist = True
        main(config)
