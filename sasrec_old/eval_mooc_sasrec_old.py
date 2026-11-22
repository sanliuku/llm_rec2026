"""sasrec_old 评估：按 run 恢复 config，加载 checkpoint，在 val/test 上跑两套指标。

用法（无命令行参数，方便批量）::

    from eval_mooc_sasrec_old import main
    main(run)                     # 评估该 run 下所有 best_sasrec_old*.pth
    main(run, ckpt='path/to/x.pth')  # 指定 ckpt
    main(run, shuffle=True)       # 打乱历史 concept 序列评估，结果单独存一份

直接运行脚本时会批量评估 sasrec_old/runs/ 下的所有 run。

两套评估体系：
1. 1-adj 掩码 + truth 位置置 1 的 per-query top-20 -> mrr / hr / ndcg；
2. 全局 top-k（屏蔽训练期已点击）-> precision / recall / f1 @5,10,20。
产物写到 run 目录：eval_<ckpt>[_shuffle].json、top20_<ckpt>[_shuffle]_*.pkl。
"""

import glob
import json
import os
import pickle as pkl
import sys
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

from config import Config, iter_run_dirs
from mooc_sasrec_old import SASRecOld
from train_mooc_sasrec_old import load_data, seed_everything
from utils import calculate, shuffle_padded_seq


def resolve_ckpts(run_dir, ckpt=None):
    """确定要评估的 checkpoint 列表：指定 ckpt / 默认该 run 下全部。"""
    if ckpt:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f'checkpoint not found: {ckpt}')
        return [ckpt]
    candidates = sorted(glob.glob(os.path.join(run_dir, 'best_sasrec_old*.pth')))
    if not candidates:
        raise FileNotFoundError(f'no checkpoint in {run_dir}')
    return candidates


def build_model(config, data, device):
    model_config = SimpleNamespace(
        concept_num=data['num_concept'],
        emb_dim=config.emb_dim,
        max_len=config.max_len,
        dropout_rate=config.dropout_rate,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        device=config.device,
    )
    return SASRecOld(model_config).to(device)


def build_log_seqs(data, config, split, shuffle=False):
    seq_dict = data['val_seq_dict'] if split == 'val' else data['test_seq_dict']
    seqs = []
    for uid in range(data['num_user']):
        seq = seq_dict.get(uid, np.zeros([config.max_len], dtype=np.int32))
        if shuffle:
            seq = shuffle_padded_seq(seq)
        seqs.append(seq.reshape(1, -1))
    return torch.from_numpy(np.concatenate(seqs, axis=0)).to(config.device)


def precision_recall(score, data, ks=(5, 10, 20)):
    """全局 top-k 口径：屏蔽训练期已点击的 concept 后，对全部 concept 直接排序取 top-k。

    truth = test 期间新点击、且训练期未点击的 concept（ua - adj）。
    """
    num_user = data['num_user']
    train_uk = data['train_val_uk_matrix'].numpy()
    test_ua = data['total_uk_matrix'].numpy() - train_uk

    truth_by_user = defaultdict(set)
    rows, cols = np.nonzero(test_ua)
    for u, c in zip(rows, cols):
        truth_by_user[int(u)].add(int(c))

    pred_score = score.numpy() * (1 - train_uk)
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


def eval_ckpt(config, data, run_dir, ckpt, device, shuffle=False):
    """评估单个 checkpoint，结果与 top-20 item 都写到 run 目录。

    shuffle=True 时用打乱后的历史序列评估，产物文件名带 _shuffle 后缀，可与默认评估并存。
    """
    ckpt_stem = os.path.splitext(os.path.basename(ckpt))[0]
    suffix = '_shuffle' if shuffle else ''
    print(f'ckpt    : {ckpt}')
    print(f'shuffle : {shuffle}')
    model = build_model(config, data, device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    result = {
        'run_id': config.run_id,
        'ckpt': os.path.basename(ckpt),
        'shuffle': shuffle,
        'config': config.to_dict(),
    }
    with torch.no_grad():
        for split in ('val', 'test'):
            log_seqs = build_log_seqs(data, config, split, shuffle=shuffle)
            score = model.predict_concept(log_seqs)
            score = torch.nn.Sigmoid()(score).cpu()
            matrix = data['train_val_uk_matrix'] if split == 'val' else data['total_uk_matrix']
            negative = data['negative'] if split == 'val' else data['test_negative']
            print(f'{split}:')
            metrics, topk_items, truth_labels = calculate(score[:, 1:], matrix, negative,
                                                          return_items=True)
            result[split] = metrics

            # 体系一：1-adj 掩码后的 per-query top-20
            path = os.path.join(run_dir, f'top20_{ckpt_stem}{suffix}_{split}_masked.pkl')
            with open(path, 'wb') as f:
                pkl.dump({'topk_items': topk_items, 'truth_labels': truth_labels}, f)
            print(f'saved: {path}')

            if split == 'test':
                # 体系二：全局 top-k（precision / recall / f1）
                print('global top-k:')
                pr_metrics, global_top20, truth_by_user = precision_recall(score[:, 1:], data)
                result['test_global'] = pr_metrics
                path = os.path.join(run_dir, f'top20_{ckpt_stem}{suffix}_test_global.pkl')
                with open(path, 'wb') as f:
                    pkl.dump({'topk_items': global_top20.tolist(),
                              'truth_by_user': {u: sorted(v) for u, v in truth_by_user.items()}}, f)
                print(f'saved: {path}')

    out = os.path.join(run_dir, f'eval_{ckpt_stem}{suffix}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'saved: {out}')
    return result


def main(run, ckpt=None, shuffle=None):
    """评估一个 run：run 可以是 run_id / run 目录 / config.json 路径。

    shuffle: None 表示沿用 config.shuffle_seq；True/False 强制用打乱/不打乱的历史序列评估，
    评估目录始终是训练时的 run 目录（shuffle 只影响序列和产物后缀）。
    """
    config = Config.load(run)
    shuffle = config.shuffle_seq if shuffle is None else bool(shuffle)
    eval_config = replace(config, shuffle_seq=shuffle)
    seed_everything(config.seed)

    device = torch.device(config.device)
    run_dir = config.run_dir()
    ckpts = resolve_ckpts(run_dir, ckpt)
    print(f'run dir : {run_dir}')
    print(f'ckpts   : {len(ckpts)}')
    print(f'shuffle : {shuffle}')

    data = load_data(eval_config)
    results = {}
    for path in ckpts:
        results[os.path.basename(path)] = eval_ckpt(eval_config, data, run_dir, path, device,
                                                    shuffle=shuffle)
    return results


if __name__ == '__main__':
    # 批量评估 runs/<dataset>/results/ 下的所有 run
    for run in iter_run_dirs():
        print(f'===== {run} =====')
        main(run)
