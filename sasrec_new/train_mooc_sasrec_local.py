import hashlib
import json
import math
import os
import pickle as pkl
import random
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

from config import Config
from mooc_sasrec import MOOCSASRec
from utils import SequenceAugDataset, BPRLoss, shuffle_padded_seq


class _Tee:
    """把 stdout/stderr 同时写到终端和 run 目录下的日志文件。"""

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


def load_data(config):
    path = os.path.join(PROJECT_ROOT, config.dataset_dir, 'need_data') + '/'

    train_val_uv_matrix = np.load(path + 'matrix/train_val_uv.npy')
    padding_matrix = np.ones((train_val_uv_matrix.shape[0], 1))
    train_val_uv_matrix = np.concatenate([padding_matrix, train_val_uv_matrix], axis=1)  # tv
    train_uv_matrix = np.load(path + 'matrix/train_uv.npy')
    train_uv_matrix = np.concatenate([padding_matrix, train_uv_matrix], axis=1)  # t

    train_val_uk_matrix = np.load(path + 'matrix/adj.npy')
    train_val_uk_matrix = torch.from_numpy(train_val_uk_matrix).float()

    total_uk_matrix = torch.from_numpy(np.load(path + 'matrix/ua.npy'))
    negative = pkl.load(open(path + 'negatives/negative.p', 'rb'))
    test_negative = pkl.load(open(path + 'negatives/test_negative.p', 'rb'))

    vk_matrix = torch.from_numpy(np.load(path + 'matrix/vk.npy')).float()
    O1 = torch.zeros((vk_matrix.size(0), vk_matrix.size(0)))
    hi = torch.cat((O1, vk_matrix), dim=1)
    O2 = torch.zeros((vk_matrix.size(1), vk_matrix.size(1)))
    lo = torch.cat((vk_matrix.T, O2), dim=1)
    L_vk = torch.cat((hi, lo), dim=0).float().to_sparse_coo()  # uv -> isomorphism

    vk_dict = pkl.load(open(path + 'relations/vk.p', 'rb'))

    # load user, video, concept data
    columns = ['index', 'id']
    user_df = pd.read_csv(path + 'entities/users.csv', header=None)
    user_df.columns = columns
    user2index = dict(zip(user_df['id'], user_df['index']))
    video_df = pd.read_csv(path + 'entities/videos.csv', header=None)
    video_df.columns = columns
    video2index = dict(zip(video_df['id'], video_df['index']))
    concept_df = pd.read_csv(path + 'entities/concepts.csv', header=None)
    concept_df.columns = columns
    concept2index = dict(zip(concept_df['id'], concept_df['index']))

    num_user = len(user_df)
    num_video = len(video_df)
    num_concept = len(concept_df)

    filtered_df = pkl.load(open(os.path.join(PROJECT_ROOT, config.dataset_dir, 'filtered_second_df2.pkl'), 'rb'))
    filtered_df['id'] = filtered_df['id'].apply(lambda x: user2index[x])
    filtered_df['vid'] = filtered_df['vid'].apply(lambda x: video2index[x])
    filtered_df = filtered_df.sort_values(by=['id', 'local_start_time'])
    train_val_df = filtered_df[(filtered_df["local_start_time"] < config.test_date_start) &
                               (filtered_df["local_start_time"] >= config.train_date_time)]
    test_df = filtered_df[(filtered_df['local_start_time'] < config.test_date_end) &
                          (filtered_df["local_start_time"] >= config.test_date_start)]
    print('loading data start....')

    # vk v+1 k+1
    new_vk_dict = {}
    for vid in vk_dict:
        new_vk_dict[vid + 1] = [concept + 1 for concept in vk_dict[vid]]

    # uv
    train_uv = {}
    train_uk = {}
    label_uv = {}
    train_val_uv = {}
    max_len = config.max_len
    for uid in train_val_df['id'].unique():
        udf = train_val_df[train_val_df['id'] == uid]
        train_val_uv[uid] = [vid + 1 for vid in udf['vid'].values.tolist()]  # train + val  -1 val -2 train  v->shift
        user_seqs = train_val_uv[uid][:-2]  # pre_seqs -2
        assert train_val_uv[uid][-2] not in user_seqs
        t = len(user_seqs) - max_len  # 49 50 51 52
        if t <= 0:
            train_uv[str(uid) + '|' + str(0)] = user_seqs
            train_uk[str(uid) + '|' + str(0)] = list(set([concept for vid in user_seqs for concept in new_vk_dict[vid]]))
            label_uv[str(uid) + '|' + str(0)] = [train_val_uv[uid][-2]]
        else:
            for i in range(t + 1):  # 0~t
                train_uv[str(uid) + '|' + str(i)] = user_seqs[:i + max_len]  # t = 1, seqs = max_len+1, label=[[max_len], [max_len+1]]
                train_uk[str(uid) + '|' + str(i)] = list(set([concept for vid in user_seqs[:i + max_len] for concept in new_vk_dict[vid]]))
                label_uv[str(uid) + '|' + str(i)] = [train_val_uv[uid][i + max_len]]
                assert train_val_uv[uid][i + max_len] != train_val_uv[uid][-1]  # val video

    test_uv = {}
    for uid in test_df['id'].unique():
        udf = test_df[test_df['id'] == uid]
        udf = udf.sort_values('local_start_time')
        test_uv[uid] = [udf['vid'].values.tolist()[0] + 1]  # v->shift

    print('loading dataset...')
    dataset = SequenceAugDataset(num_i=num_video, num_k=num_concept, ua_dict=train_uv, label_dict=label_uv,
                                 uk_dict=train_uk, vk_dict=new_vk_dict, max_len=max_len, shuffle_seq=config.shuffle_seq)
    # val not in train
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    print('load data done')

    return {
        'path': path,
        'train_val_uv_matrix': train_val_uv_matrix,
        'train_uv_matrix': train_uv_matrix,
        'train_val_uk_matrix': train_val_uk_matrix,
        'total_uk_matrix': total_uk_matrix,
        'negative': negative,
        'test_negative': test_negative,
        'L_vk': L_vk,
        'new_vk_dict': new_vk_dict,
        'num_user': num_user,
        'num_video': num_video,
        'num_concept': num_concept,
        'train_val_uv': train_val_uv,
        'test_uv': test_uv,
        'dataloader': dataloader,
    }


def save_prepared_data(config, data):
    """把本次训练用的数据存到 <save_root>/<dataset>/prepare_data/，文件名按数据相关参数区分。"""
    payload = {
        'dataset_dir': config.dataset_dir,
        'train_date_time': config.train_date_time,
        'test_date_start': config.test_date_start,
        'test_date_end': config.test_date_end,
        'max_len': config.max_len,
    }
    data_id = hashlib.md5(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()
    prepare_dir = config.prepare_data_dir(PROJECT_ROOT)
    os.makedirs(prepare_dir, exist_ok=True)
    path = os.path.join(prepare_dir, f'prepared_{data_id}.pkl')
    with open(path, 'wb') as f:
        pkl.dump({
            'data_config': payload,
            'num_user': data['num_user'],
            'num_video': data['num_video'],
            'num_concept': data['num_concept'],
            'train_val_uv': data['train_val_uv'],
            'test_uv': data['test_uv'],
            'new_vk_dict': data['new_vk_dict'],
        }, f)
    print(f'prepared data saved: {path}')
    return path


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


def calculate(score, adj, test_negative, is_val=True, return_items=False):
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
        # 只需要 top-10，用 argpartition 避免对全部 1.5w 个候选做完整排序
        candidate = np.argpartition(-score_new, topk - 1, axis=1)[:, :topk]
        candidate_scores = np.take_along_axis(score_new, candidate, axis=1)
        order = np.argsort(-candidate_scores, axis=1)
        topk_items_sub = np.take_along_axis(candidate, order, axis=1).tolist()
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
    metrics = {
        'mrr': mrr,
        'hr@1': hr_1, 'hr@5': hr_5, 'hr@10': hr_10, 'hr@20': hr_20,
        'ndcg@1': ndcg_1, 'ndcg@5': ndcg_5, 'ndcg@10': ndcg_10, 'ndcg@20': ndcg_20,
    }
    if return_items:
        # topk_items: 每条 (user, truth) 查询对应的 masked top-20 concept id
        return metrics, topk_items, truth_labels
    return metrics


def train(model, dataloader, loss_fn, alpha, config, device, optimizer):
    model.train()
    loss_mean = 0
    for u, seq, pos, neg, pos_concepts, neg_concepts in dataloader:
        seq, pos, neg, pos_concepts, neg_concepts = seq.to(device), pos.to(device), neg.to(device), pos_concepts.to(device), neg_concepts.to(device)
        optimizer.zero_grad()
        pos_logits, neg_logits, pos_kpred, neg_kpred = model(seq, pos, neg, pos_concepts, neg_concepts)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=device), torch.zeros(neg_logits.shape, device=device)
        indices = torch.where(pos != 0)
        loss = alpha * (loss_fn(pos_logits[indices], pos_labels[indices]) + loss_fn(neg_logits[indices], neg_labels[indices]))

        kindices = torch.where(pos_concepts != 0)
        loss += (1 - alpha) * BPRLoss(pos_kpred[kindices], neg_kpred[kindices])

        for param in model.video_emb.parameters():
            loss += config.weight_decay * torch.norm(param)
        loss.backward()
        optimizer.step()
        loss_mean += loss.item() * seq.size(0)
    loss_mean /= len(dataloader.dataset)
    return loss_mean


def build_log_seqs(uv, config, drop_last_video=False):
    """把每个用户的历史 video 序列左 padding 到 max_len。val 去尾，test 用全量。"""
    log_seqs = []
    for uid in range(len(uv)):
        idx = 0
        seq = np.zeros([config.max_len], dtype=np.int32)
        log_seq = uv[uid][:-1] if drop_last_video else uv[uid]
        for t in reversed(log_seq):
            seq[config.max_len - idx - 1] = t
            idx += 1
            if idx == config.max_len:
                break
        if config.shuffle_seq:
            seq = shuffle_padded_seq(seq)
        log_seqs.append(seq.reshape(1, -1))
    log_seqs = np.concatenate(log_seqs, axis=0)
    return torch.from_numpy(log_seqs)


def val(model, data, config, device):
    log_seqs = build_log_seqs(data['train_val_uv'], config, drop_last_video=True).to(device)
    score = model.predict_concept(log_seqs)
    score = torch.nn.Sigmoid()(score).cpu().detach()
    return calculate(score[:, 1:], data['train_val_uk_matrix'], data['negative'])['mrr']


def eval(model, data, config, device):
    log_seqs = build_log_seqs(data['train_val_uv'], config).to(device)
    score = model.predict_concept(log_seqs)
    score = torch.nn.Sigmoid()(score).cpu().detach()
    return calculate(score[:, 1:], data['total_uk_matrix'], data['test_negative'])['mrr']


def main(config: Config):
    seed_everything(config.seed)

    run_dir = config.create_run_dir(PROJECT_ROOT)
    log_file = open(os.path.join(run_dir, 'train.log'), 'a', encoding='utf-8')
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    metrics_file = open(os.path.join(run_dir, 'metrics.jsonl'), 'a', encoding='utf-8')

    device = torch.device(config.device)
    print(f'run dir: {run_dir}')

    data = load_data(config)
    # save_prepared_data(config, data)

    model_config = SimpleNamespace(
        user_num=data['num_user'],
        video_num=data['num_video'],
        concept_num=data['num_concept'],
        bert_dim=config.bert_dim,
        emb_dim=config.emb_dim,
        max_len=config.max_len,
        dropout_rate=config.dropout_rate,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        device=device,
    )

    print('load model')
    bce_criterion = torch.nn.BCEWithLogitsLoss()

    for alpha in reversed(config.alphas):
        init_index = 0
        max_val = 0
        best_mrr = 0

        seed_everything(config.seed)
        model = MOOCSASRec(model_config, edge_index=data['L_vk'].indices(),
                           emb_path=data['path'] + 'embs/concept_embs_local.npy').to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        for epoch in range(config.epochs):
            train_loss = train(model, data['dataloader'], bce_criterion, alpha, config, device, optimizer)
            with torch.no_grad():
                model.eval()
                val_mrr = val(model, data, config, device)
                if epoch % config.eval_interval == 0:
                    best_mrr = eval(model, data, config, device)
                if val_mrr > max_val:
                    torch.save(model.state_dict(),
                               os.path.join(run_dir, f'best_sasrec_mooc{str(int(alpha * 10))}.pth'))
                    max_val = val_mrr
                    init_index = 0
                else:
                    init_index += 1
                print(f'alpha: {alpha}, epoch: {epoch}, train_loss: {train_loss}, '
                      f'val_mrr: {val_mrr}, mrr: {best_mrr}')
                metrics_file.write(json.dumps({
                    'alpha': alpha,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_mrr': val_mrr,
                    'test_mrr': best_mrr,
                }, ensure_ascii=False) + '\n')
                metrics_file.flush()
                if init_index > config.early_stop:
                    print(f'early stop at epoch {epoch}')
                    break
    print('train done')

    metrics_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()


if __name__ == '__main__':
    config = Config()
    dataset_dir = "data/MOOCCube/small_data"
    train_date_time = '2019-01-01 00:00:00'
    test_date_start = '2019-05-01 00:00:00'
    test_date_end = '2019-06-01 00:00:00'
    alphas = (0.7, )

    config.dataset_dir = dataset_dir
    config.train_date_time = train_date_time
    config.test_date_start = test_date_start
    config.test_date_end = test_date_end
    config.alphas = alphas

    seeds = [12, 22, 32, 52]
    for seed in seeds:
        config.seed = seed
        main(config)
