"""A-LLMRec 第一阶段的公共组件：CF 模型加载、文本编码器加载、评测与损失统计。

数据构造在 `align_data.py`，这里只管「模型怎么搭、分怎么打」。
"""

import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
for _p in (PROJECT_ROOT, CURRENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sasrec_old.mooc_sasrec_old import SASRecOld
from sasrec_old.utils import calculate
from text_encoder import ConceptTextEncoder, PrecomputedTextEncoder, load_concept_texts


def resolve(path: str) -> str:
    """config 里的相对路径都相对项目根目录。"""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(PROJECT_ROOT, path))


def load_cf_config(config) -> dict:
    """读冻结 CF 那个 run 的 config.json（数据口径以它为准）。"""
    with open(os.path.join(config.cf_run_dir(PROJECT_ROOT), 'config.json'), encoding='utf-8') as f:
        return json.load(f)


def load_cf_model(config, device) -> tuple:
    """读 sasrec_old 一个 run 的 config.json + ckpt，构建冻结的 CF 模型。

    结构参数取自该 run 的 config.json，concept 行数以 ckpt 实际形状为准。
    """
    run_dir = config.cf_run_dir(PROJECT_ROOT)
    with open(os.path.join(run_dir, 'config.json'), encoding='utf-8') as f:
        cf = json.load(f)
    state = torch.load(os.path.join(run_dir, config.cf_ckpt), map_location='cpu', weights_only=True)

    cf_config = SimpleNamespace(
        concept_num=state['concept_emb.weight'].shape[0] - 1,
        emb_dim=cf['emb_dim'],
        max_len=cf['max_len'],
        dropout_rate=cf['dropout_rate'],
        num_heads=cf['num_heads'],
        num_blocks=cf['num_blocks'],
        device=str(device),
    )
    model = SASRecOld(cf_config).to(device)
    model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f'CF model: {run_dir} (concept_num={cf_config.concept_num}, '
          f'emb_dim={cf_config.emb_dim}, max_len={cf_config.max_len})')
    return model, cf_config


def load_text_model(config, num_concept: int, device):
    """构建文本侧 Q_i：给了 text_emb_path 就用离线表，否则在线跑 SBERT。"""
    if config.text_emb_path:
        encoder = PrecomputedTextEncoder(resolve(config.text_emb_path), num_concept).to(device)
        print('  -> Q_i 用离线表，text_finetune / sbert_freeze_layers / sbert_lr 均不生效')
        return encoder

    texts = load_concept_texts(resolve(config.concept_csv), resolve(config.text_pkl), num_concept)
    encoder = ConceptTextEncoder(resolve(config.text_model), texts,
                                 max_length=config.text_max_length,
                                 finetune=config.text_finetune,
                                 freeze_layers=config.sbert_freeze_layers,
                                 freeze_embeddings=config.freeze_embeddings).to(device)
    if encoder.emb_dim != config.text_dim:
        raise ValueError(f'text_dim={config.text_dim} 与 {config.text_model} 的 {encoder.emb_dim} 不一致')
    return encoder


def full_scores(model, seqs: np.ndarray, device, batch_size: int = 1000) -> np.ndarray:
    """(num_user, max_len) 的序列 -> (num_user, num_concept) 分数（sigmoid 后去掉 padding 列）。

    与 sasrec_old.evaluate 一致：先过 Sigmoid 再排序，并列在严格 `>` 下不计入 better。
    """
    scores = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            log_seqs = torch.from_numpy(seqs[i:i + batch_size]).to(device)
            score = torch.nn.Sigmoid()(model.predict_concept(log_seqs))
            scores.append(score.cpu().numpy()[:, 1:])
    return np.concatenate(scores, axis=0)


def evaluate(model, seqs: np.ndarray, adj: np.ndarray, negative: dict, device,
             return_items: bool = False):
    """直接套用 sasrec_old.utils.calculate()，指标口径与 CF 训练日志逐位一致。

    seqs    : (num_user, max_len)，每用户一条历史
    adj     : val 用 adj.npy（train+val 交互），test 用 ua.npy（全交互）
    negative: {uid: [[neg..., truth], ...]}，末位是 0-based 的 truth 列号
    return_items=True 时额外返回 (metrics, topk_items, truth_labels)。
    """
    score = full_scores(model, seqs, device)
    return calculate(score, adj, negative, return_items=return_items)


def precision_recall(score, train_uk: np.ndarray, test_uk: np.ndarray, ks=(5, 10, 20)):
    """体系二：全局 top-k 口径（对应 sasrec_old.eval_mooc_sasrec_old.precision_recall）。

    屏蔽训练期已点击的 concept 后，对全部 concept 直接排序取 top-k；
    truth = test 期间新点击、训练期未点击的 concept（test_uk & ~train_uk）。
    返回 (metrics, top20_items, truth_by_user)。
    """
    score = score.numpy() if torch.is_tensor(score) else np.asarray(score)
    train_uk = np.asarray(train_uk).astype(bool)
    test_uk = np.asarray(test_uk).astype(bool)
    num_user = score.shape[0]

    truth_by_user = defaultdict(set)
    rows, cols = np.nonzero(test_uk & ~train_uk)
    for u, c in zip(rows, cols):
        truth_by_user[int(u)].add(int(c))

    pred_score = score * (1 - train_uk)
    sorted_items = np.argsort(-pred_score, axis=1)

    metrics = {}
    for k in ks:
        precisions, recalls = [], []
        topk_items = sorted_items[:, :k]
        for i in range(num_user):
            hits = len(set(topk_items[i].tolist()) & truth_by_user[i])
            precisions.append(hits / k)
            recalls.append(hits / len(truth_by_user[i]) if truth_by_user[i] else 0.0)
        p = float(np.mean(precisions))
        r = float(np.mean(recalls))
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        metrics[f'precision@{k}'] = p
        metrics[f'recall@{k}'] = r
        metrics[f'f1@{k}'] = f1
        print(f'precision@{k}: {p}, recall@{k}: {r}, f1@{k}: {f1}')
    return metrics, sorted_items[:, :20], truth_by_user


def cf_baseline(cf_model, seqs: np.ndarray, adj: np.ndarray, negative: dict, device) -> dict:
    """冻结 CF 自己（不经过对齐 MLP）在同一份数据上的指标，作为 stage-1 的对照。"""
    return evaluate(cf_model, seqs, adj, negative, device)
