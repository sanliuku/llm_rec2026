"""A-LLMRec 第一阶段评估：按 run 恢复 config，加载 checkpoint，在 val/test 上跑全量排名指标。














"""

import glob
import json
import os
import pickle as pkl
import sys

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

from align_utils import calculate, full_scores, precision_recall
from config_align import AlignConfig, iter_run_dirs
from mooc_align import MoocAlign
from train_mooc_align import load_data, seed_everything


def resolve_ckpts(run_dir, ckpt=None):
    """确定要评估的 checkpoint 列表：指定 ckpt / 默认该 run 下的 best_align.pth。"""
    if ckpt:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f'checkpoint not found: {ckpt}')
        return [ckpt]
    candidates = sorted(glob.glob(os.path.join(run_dir, 'best_align*.pth')))
    if not candidates:
        raise FileNotFoundError(f'no checkpoint in {run_dir}')
    return candidates


def align_cosine(model) -> float:
    """对齐空间里联合嵌入 e_i 与文本嵌入 q_i 的平均余弦（只统计非 padding 行）。"""
    with torch.no_grad():
        e = model.joint_item_embeddings()[1:]
        q = model.joint_text_embeddings()[1:]
        cos = torch.nn.functional.cosine_similarity(e, q, dim=-1)
    return float(cos.mean())


def eval_ckpt(config, data, run_dir, ckpt, device):
    """评估单个 checkpoint，结果写到 run 目录。"""
    ckpt_stem = os.path.splitext(os.path.basename(ckpt))[0]
    print(f'ckpt    : {ckpt}')
    model = MoocAlign(data['cf_model'], data['text_model'], config).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    result = {
        'run_id': config.run_id,
        'ckpt': os.path.basename(ckpt),
        'config': config.to_dict(),
        'align_cosine': align_cosine(model),
    }
    print(f'align cosine (e_i, q_i): {result["align_cosine"]:.4f}')

    for split in ('val', 'test'):
        print(f'{split}:')
        # 掩码与 sasrec_old 一致：val -> adj.npy（train+val），test -> ua.npy（全交互）
        if split == 'val':
            seqs, matrix, negative = data['val_seqs'], data['adj'], data['negative']
        else:
            seqs, matrix, negative = data['test_seqs'], data['ua'], data['test_negative']

        score = full_scores(model, seqs, device)
        # 体系一：1-adj 掩码后的 per-query top-20（mrr / hr / ndcg）
        metrics, topk_items, truth_labels = calculate(score, matrix, negative, return_items=True)
        base = data['baseline'][split]['mrr']
        print(f'  baseline(冻结 CF) mrr: {base:.5f} | 本 ckpt mrr: {metrics["mrr"]:.5f} '
              f'({(metrics["mrr"] / base - 1) * 100:+.1f}%)')
        result[split] = metrics
        result[f'{split}_baseline'] = data['baseline'][split]

        path = os.path.join(run_dir, f'top20_{ckpt_stem}_{split}_masked.pkl')
        with open(path, 'wb') as f:
            pkl.dump({'topk_items': topk_items, 'truth_labels': truth_labels}, f)
        print(f'saved: {path}')

        if split == 'test':
            # 体系二：全局 top-k（屏蔽训练期点击），truth = ua - adj -> precision / recall / f1
            print('global top-k:')
            pr_metrics, global_top20, truth_by_user = precision_recall(score, data['adj'], data['ua'])
            result['test_global'] = pr_metrics
            path = os.path.join(run_dir, f'top20_{ckpt_stem}_test_global.pkl')
            with open(path, 'wb') as f:
                pkl.dump({'topk_items': global_top20.tolist(),
                          'truth_by_user': {u: sorted(v) for u, v in truth_by_user.items()}}, f)
            print(f'saved: {path}')

    out = os.path.join(run_dir, f'eval_{ckpt_stem}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)
    print(f'saved: {out}')
    return result


def main(run, ckpt=None):
    """评估一个 run：run 可以是 run_id / run 目录 / config.json 路径。"""
    config = AlignConfig.load(run)
    seed_everything(config.seed)

    device = torch.device(config.device)
    run_dir = config.run_dir()
    ckpts = resolve_ckpts(run_dir, ckpt)
    print(f'run dir : {run_dir}')
    print(f'ckpts   : {len(ckpts)}')

    data = load_data(config, device)
    results = {}
    for path in ckpts:
        results[os.path.basename(path)] = eval_ckpt(config, data, run_dir, path, device)
    return results


if __name__ == '__main__':
    # 批量评估 runs/<dataset>/results/ 下的所有 run
    for run in iter_run_dirs():
        print(f'===== {run} =====')
        main(run)
