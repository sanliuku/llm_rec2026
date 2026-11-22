"""汇总 ALLMRec 各 run 的评估结果：均值、标准差与 95% 置信区间。

只读已有结果文件，不做任何重算：

1. HR@k / NDCG@k：run 目录下 ``final_result_*_match.json``
2. precision / recall / f1 @5,10：run 目录下 ``recall_result.json``；
   没有该 json 时读 ``recall_result.txt``（metric_score.py 的 stdout 记录）

目录结构::

    ALLMRec/runs/<dataset>/results/<run_id>/final_result_*_match.json
    ALLMRec/runs/<dataset>/results/<run_id>/recall_result.json
    ALLMRec/runs/<dataset>/results/aggregate_eval.json     # 本脚本产物

用法::

    python ALLMRec/aggregate_eval.py                # 默认 small_data
    python ALLMRec/aggregate_eval.py mid_data
"""

import glob
import importlib.util
import json
import os

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT_DIR)

ORDER = ('HR@1', 'HR@5', 'HR@10', 'HR@20', 'NDCG@1', 'NDCG@5', 'NDCG@10', 'NDCG@20',
         'precision@5', 'recall@5', 'f1@5', 'precision@10', 'recall@10', 'f1@10')


def _load_summary_stats():
    """复用 sasrec_new/aggregate_runs.py 的 summary_stats（避免两套统计口径）。"""
    path = os.path.join(ROOT, 'sasrec_new', 'aggregate_runs.py')
    spec = importlib.util.spec_from_file_location('sasrec_aggregate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summary_stats


summary_stats = _load_summary_stats()


def run_dirs(dataset):
    base = os.path.join(CURRENT_DIR, 'runs', dataset, 'results')
    if not os.path.isdir(base):
        raise FileNotFoundError(f'没有这个数据集的结果目录: {base}')
    return [d for d in sorted(glob.glob(os.path.join(base, '*'))) if os.path.isdir(d)]


def tag_of(path):
    """final_result_<tag>_match.json -> <tag>"""
    return os.path.basename(path)[len('final_result_'):-len('_match.json')]


def read_rank_metrics(path):
    """final_result_*_match.json 里的 HR@k / NDCG@k。"""
    with open(path, encoding='utf-8') as f:
        return {k: float(v) for k, v in json.load(f).items() if k in ORDER}


def read_retrieval_metrics(run_dir):
    """recall_result.json（优先）或 recall_result.txt 里的 precision / recall / f1。"""
    path = os.path.join(run_dir, 'recall_result.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            metrics = {k: float(v) for k, v in json.load(f).items() if k in ORDER}
        if metrics:
            return metrics, 'recall_json'

    path = os.path.join(run_dir, 'recall_result.txt')
    if os.path.exists(path):
        metrics = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                name, _, value = line.strip().partition(':')
                name = name.strip()
                if name in ORDER:
                    try:
                        metrics[name] = float(value.strip())
                    except ValueError:
                        continue
        if metrics:
            return metrics, 'recall_txt'
    return {}, None


def aggregate(dataset='small_data', confidence=0.95):
    grouped, used, skipped, sources = {}, [], [], {}

    for run_dir in run_dirs(dataset):
        name = os.path.basename(run_dir)
        rank_files = sorted(glob.glob(os.path.join(run_dir, 'final_result_*_match.json')))
        retrieval, source = read_retrieval_metrics(run_dir)
        if not rank_files and not retrieval:
            skipped.append(name)
            continue
        used.append(name)

        for path in rank_files:
            for metric, value in read_rank_metrics(path).items():
                grouped.setdefault(tag_of(path), {}).setdefault(metric, []).append(value)
        if retrieval:
            tag = tag_of(rank_files[0]) if rank_files else 'ALL'
            for metric, value in retrieval.items():
                grouped.setdefault(tag, {}).setdefault(metric, []).append(value)
        sources[name] = source

    if not grouped:
        raise FileNotFoundError(f'{dataset} 下没有可汇总的结果文件')

    payload = {'dataset': dataset, 'confidence': confidence,
               'run_count': len(used), 'runs': used, 'skipped_runs': skipped,
               'retrieval_sources': sources, 'ckpts': {}}

    for tag, metrics in sorted(grouped.items()):
        names = [n for n in ORDER if n in metrics] + sorted(set(metrics) - set(ORDER))
        print(f'\n===== {dataset}  {tag}  (runs: {len(used)}, {confidence:.0%} CI) =====')
        print(f'{"metric":<14s}{"n":>3s}  {"mean":>9s} {"std":>9s}   {"95% CI":>19s}')
        payload['ckpts'][tag] = {}
        for metric in names:
            s = summary_stats(metrics[metric], confidence)
            payload['ckpts'][tag][metric] = s
            ci = f'[{s["ci_low"]:.5f}, {s["ci_high"]:.5f}]'
            print(f'{metric:<14s}{s["n"]:>3d}  {s["mean"]:>9.5f} {s["std"]:>9.5f}   {ci:>19s}')

    if skipped:
        print(f'\n跳过（无结果文件）: {skipped}')
    print('\n数据来源:')
    for name in used:
        print(f'  {name[:8]}  {sources.get(name)}')

    out = os.path.join(CURRENT_DIR, 'runs', dataset, 'results', 'aggregate_eval.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)
    print(f'\nsaved: {out}')
    return payload


if __name__ == '__main__':
    import sys
    aggregate(sys.argv[1] if len(sys.argv) > 1 else 'mid_data')
