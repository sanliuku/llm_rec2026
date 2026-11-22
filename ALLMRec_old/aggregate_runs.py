"""跨 run 汇总 stage-1 的评估结果：均值、标准差与 95% 置信区间。

目录结构::

    runs/<dataset>/prepare_data/
    runs/<dataset>/results/<md5>/          # 每个 run 的 eval_*.json
    runs/<dataset>/results/aggregate.json  # 汇总结果

eval_*.json 由 eval_mooc_align.py 产出，字段：ckpt / align_cosine / val / test /
val_baseline / test_baseline（baseline 是冻结 CF 在同一份数据上的数值，默认不汇总）。

用法::

    python aggregate_runs.py                     # 按数据集分别汇总所有 run
    from aggregate_runs import aggregate_runs
    aggregate_runs([run1, run2], confidence=0.95)
"""

import glob
import json
import os
from collections import defaultdict

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def summary_stats(values, confidence=0.95):
    """一组数的 均值、标准差（ddof=1）和 95% 置信区间（t 分布，双尾）。"""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    half = 0.0
    if n > 1:
        try:
            from scipy.stats import t
            crit = float(t.ppf(0.5 + confidence / 2, n - 1))
        except ImportError:  # 没有 scipy 时用正态近似
            crit = 1.96
        half = crit * std / np.sqrt(n)
    return {
        'n': n,
        'mean': mean,
        'std': std,
        'ci_low': mean - half,
        'ci_high': mean + half,
    }


def aggregate_runs(runs, confidence=0.95, save_path=None, with_baseline=False):
    """跨多个 run 汇总 eval 结果：按 ckpt 名分组，输出 均值±标准差 与置信区间。

    runs: run 目录列表。
    save_path: 结果保存路径；None 时保存到这些 run 所属的 results 目录下的 aggregate.json。
    with_baseline: 是否一并汇总 val_baseline / test_baseline（冻结 CF 基线，跨 run 相同）。
    返回 {ckpt: {split: {metric: {...}}}}，align_cosine 归到 split='align' 下。
    """
    runs = [d for d in runs if os.path.isdir(d)]
    if save_path is None and runs:
        save_path = os.path.join(os.path.dirname(os.path.normpath(runs[0])), 'aggregate.json')

    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    runs_seen = defaultdict(set)
    missing = []
    for run in runs:
        paths = sorted(glob.glob(os.path.join(run, 'eval_*.json')))
        if not paths:
            missing.append(run)
            continue
        for path in paths:
            with open(path, encoding='utf-8') as f:
                res = json.load(f)
            ckpt = res.get('ckpt') or os.path.basename(path)[len('eval_'):-len('.json')]
            runs_seen[ckpt].add(os.path.basename(os.path.normpath(run)))

            splits = ['val', 'test', 'test_global']
            if with_baseline:
                splits += ['val_baseline', 'test_baseline']
            for split in splits:
                for metric, value in (res.get(split) or {}).items():
                    if isinstance(value, (int, float)):
                        grouped[ckpt][split][metric].append(float(value))
            if isinstance(res.get('align_cosine'), (int, float)):
                grouped[ckpt]['align']['cosine'].append(float(res['align_cosine']))

    if missing:
        print(f'\n没有 eval_*.json 的 run（{len(missing)} 个）:')
        for run in missing:
            print(f'  {run}')

    if not grouped:
        raise FileNotFoundError('no eval_*.json found, 先跑 eval_mooc_align 生成评估结果')

    summary = {}
    for ckpt, splits in grouped.items():
        summary[ckpt] = {}
        print(f'\n===== {ckpt}  (runs: {len(runs_seen[ckpt])}, {confidence:.0%} CI) =====')
        for split, metrics in splits.items():
            summary[ckpt][split] = {}
            print(f'[{split}]')
            for metric, values in metrics.items():
                s = summary_stats(values, confidence)
                summary[ckpt][split][metric] = s
                print(f'  {metric:14s} {s["mean"]:.5f} ± {s["std"]:.5f}   '
                      f'95% CI [{s["ci_low"]:.5f}, {s["ci_high"]:.5f}]  (n={s["n"]})')

    if save_path:
        payload = {
            'confidence': confidence,
            'run_dirs': [os.path.basename(os.path.normpath(r)) for r in runs if r not in missing],
            'run_count': len(runs) - len(missing),
            'missing_eval_runs': [os.path.basename(os.path.normpath(r)) for r in missing],
            'ckpts': summary,
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=float)
        print(f'\nsaved: {save_path}')
    return summary


def aggregate_all(confidence=0.95, with_baseline=False):
    """按数据集遍历 runs/<dataset>/results/，各自汇总并保存 aggregate.json。"""
    results_dirs = sorted(glob.glob(os.path.join(CURRENT_DIR, 'runs', '*', 'results')))
    if not results_dirs:
        print('no runs/<dataset>/results found')
        return {}
    all_summary = {}
    for results_dir in results_dirs:
        runs = [d for d in sorted(glob.glob(os.path.join(results_dir, '*'))) if os.path.isdir(d)]
        if not runs:
            continue
        dataset = os.path.basename(os.path.dirname(results_dir))
        print(f'\n######## dataset: {dataset}  ({len(runs)} runs) ########')
        all_summary[dataset] = aggregate_runs(runs, confidence=confidence,
                                              save_path=os.path.join(results_dir, 'aggregate.json'),
                                              with_baseline=with_baseline)
    return all_summary


if __name__ == '__main__':
    aggregate_all()
