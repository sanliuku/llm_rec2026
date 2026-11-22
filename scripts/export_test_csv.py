"""把 sasrec/lru 多条 run 的 aggregate 结果统一导出成 CSV（只取 test 划分）。

输出两类文件（数值保留 4 位小数）：

1. ``results/sasrec_lru_test_aggregate.csv``：完整明细，按 (模型, 数据集, ckpt, shuffle) 分行，
   每个指标给出 mean / std / 95% CI / n_runs
2. ``results/test_<dataset>_shuffle_<True|False>.csv``：拆成 4 张紧凑表（数据集 × shuffle），
   只保留「模型 + 指标」，单元格为 ``mean±std [95% CI low, high]``

汇总口径与 <module>/aggregate_runs.py 一致：均值、标准差(ddof=1)、95% CI(t 分布)。

用法::

    python scripts/export_test_csv.py
"""

import csv
import glob
import json
import os
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'sasrec_old'))

from aggregate_runs import summary_stats  # noqa: E402

MODULES = ['sasrec_old', 'sasrec_new', 'lru_new']
SPLIT = 'test'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
METRIC_ORDER = ['mrr', 'hr@1', 'hr@5', 'hr@10', 'hr@20',
                'ndcg@1', 'ndcg@5', 'ndcg@10', 'ndcg@20']


def collect():
    grouped = defaultdict(lambda: defaultdict(list))
    runs_seen = defaultdict(set)
    for module in MODULES:
        for path in sorted(glob.glob(os.path.join(PROJECT_ROOT, module, 'runs', '*', 'results', '*', 'eval_*.json'))):
            with open(path, encoding='utf-8') as f:
                res = json.load(f)
            if SPLIT not in res:
                continue
            dataset = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
            run_id = os.path.basename(os.path.dirname(path))
            key = (module, dataset, res['ckpt'], bool(res.get('shuffle', False)))
            runs_seen[key].add(run_id)
            for metric, value in res[SPLIT].items():
                grouped[key][metric].append(float(value))
    return grouped, runs_seen


def ordered(metrics):
    known = [m for m in METRIC_ORDER if m in metrics]
    return known + sorted(m for m in metrics if m not in METRIC_ORDER)


def write_detail(grouped, runs_seen, metrics):
    header = ['model', 'dataset', 'ckpt', 'shuffle', 'n_runs']
    for m in metrics:
        header += [f'{m}_mean', f'{m}_std', f'{m}_ci_low', f'{m}_ci_high']

    rows = []
    for (model, dataset, ckpt, shuffle), per_metric in grouped.items():
        stats = {m: summary_stats(v) for m, v in per_metric.items()}
        row = [model, dataset, ckpt, str(shuffle), len(runs_seen[(model, dataset, ckpt, shuffle)])]
        for m in metrics:
            s = stats.get(m)
            row += [''] * 4 if s is None else [f'{s["mean"]:.4f}', f'{s["std"]:.4f}',
                                               f'{s["ci_low"]:.4f}', f'{s["ci_high"]:.4f}']
        rows.append(row)

    rows.sort(key=lambda r: (MODULES.index(r[0]), r[1], r[3]))
    path = os.path.join(RESULTS_DIR, 'sasrec_lru_test_aggregate.csv')
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f'saved: {path}  ({len(rows)} rows)')


def write_splits(grouped, metrics):
    combos = sorted({(dataset, shuffle) for _, dataset, _, shuffle in grouped})
    for dataset, shuffle in combos:
        rows = []
        for model in MODULES:
            keys = [k for k in grouped if k[0] == model and k[1] == dataset and k[3] == shuffle]
            if not keys:
                continue
            values = defaultdict(list)
            for key in keys:
                for m, v in grouped[key].items():
                    values[m].extend(v)
            cells = []
            for m in metrics:
                s = summary_stats(values[m]) if m in values else None
                cells.append('' if s is None else
                             f'{s["mean"]:.4f}±{s["std"]:.4f} '
                             f'[{s["ci_low"]:.4f}, {s["ci_high"]:.4f}]')
            rows.append([model] + cells)

        path = os.path.join(RESULTS_DIR, f'test_{dataset}_shuffle_{shuffle}.csv')
        with open(path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model'] + metrics)
            writer.writerows(rows)
        print(f'saved: {path}  ({len(rows)} models)')


def main():
    grouped, runs_seen = collect()
    if not grouped:
        raise FileNotFoundError(f'no eval_*.json with {SPLIT!r} found under {MODULES}')

    metrics = ordered({m for per_metric in grouped.values() for m in per_metric})
    os.makedirs(RESULTS_DIR, exist_ok=True)
    write_detail(grouped, runs_seen, metrics)
    write_splits(grouped, metrics)


if __name__ == '__main__':
    main()
