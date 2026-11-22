"""汇总 LLaRA_old 各 run 的推理结果：均值、标准差与 95% 置信区间。

只读已有结果文件，不做任何重算：

1. hr@k / ndcg@k / hit_rate / miss：run 目录下 ``metrics_<cand_source>.json``
   （infer_mooc_llara_old.py 的产物，按候选口径 align_masked / sasrec_global 分组）
2. precision / recall / f1 @5,10：run 目录下 ``recall_result.txt``（recall 口径，单独一组）

目录结构::

    LLaRA_old/runs/<dataset>/results/<run_id>/metrics_align_masked.json
    LLaRA_old/runs/<dataset>/results/<run_id>/metrics_sasrec_global.json
    LLaRA_old/runs/<dataset>/results/<run_id>/recall_result.txt
    LLaRA_old/runs/<dataset>/results/aggregate_eval.json      # 本脚本产物

用法::

    python LLaRA_old/aggregate_eval.py                # 汇总 runs 下所有有结果的数据集
    python LLaRA_old/aggregate_eval.py small_data     # 只汇总一个数据集
"""

import glob
import importlib.util
import json
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT_DIR)

# metrics_*.json 里的排序指标（n 是样本数，不参与统计）
RANK_ORDER = ('hit_rate', 'hr@1', 'ndcg@1', 'hr@5', 'ndcg@5', 'hr@10', 'ndcg@10', 'miss')
# recall_result.txt 里的检索指标
RETR_ORDER = ('precision@5', 'recall@5', 'f1@5', 'precision@10', 'recall@10', 'f1@10')


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
    """metrics_<tag>.json -> <tag>"""
    return os.path.basename(path)[len('metrics_'):-len('.json')]


def read_rank_metrics(path):
    """metrics_<tag>.json 里嵌套的 metrics 字段。"""
    with open(path, encoding='utf-8') as f:
        payload = json.load(f)
    metrics = payload.get('metrics', payload)
    return {k: float(v) for k, v in metrics.items() if k in RANK_ORDER}


def read_retrieval_metrics(run_dir):
    """recall_result.json（优先）或 recall_result.txt 里的 precision / recall / f1。"""
    path = os.path.join(run_dir, 'recall_result.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            metrics = {k: float(v) for k, v in json.load(f).items() if k in RETR_ORDER}
        if metrics:
            return metrics, 'recall_json'

    path = os.path.join(run_dir, 'recall_result.txt')
    if os.path.exists(path):
        metrics = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                name, _, value = line.strip().partition(':')
                name = name.strip()
                if name in RETR_ORDER:
                    try:
                        metrics[name] = float(value.strip())
                    except ValueError:
                        continue
        if metrics:
            return metrics, 'recall_txt'
    return {}, None


def print_block(title, metrics, confidence):
    names = [n for n in (RANK_ORDER + RETR_ORDER) if n in metrics] + sorted(set(metrics) - set(RANK_ORDER) - set(RETR_ORDER))
    print(f'\n===== {title}  (runs: {len(next(iter(metrics.values())))} , {confidence:.0%} CI) =====')
    print(f'{"metric":<14s}{"n":>3s}  {"mean":>9s} {"std":>9s}   {"95% CI":>19s}')
    out = {}
    for metric in names:
        s = summary_stats(metrics[metric], confidence)
        out[metric] = s
        ci = f'[{s["ci_low"]:.5f}, {s["ci_high"]:.5f}]'
        print(f'{metric:<14s}{s["n"]:>3d}  {s["mean"]:>9.5f} {s["std"]:>9.5f}   {ci:>19s}')
    return out


def aggregate(dataset='small_data', confidence=0.95):
    grouped, used, skipped, sources = {}, [], [], {}

    for run_dir in run_dirs(dataset):
        name = os.path.basename(run_dir)
        rank_files = sorted(glob.glob(os.path.join(run_dir, 'metrics_*.json')))
        retrieval, source = read_retrieval_metrics(run_dir)
        if not rank_files and not retrieval:
            skipped.append(name)
            continue
        used.append(name)

        for path in rank_files:
            for metric, value in read_rank_metrics(path).items():
                grouped.setdefault(tag_of(path), {}).setdefault(metric, []).append(value)
        if retrieval:
            for metric, value in retrieval.items():
                grouped.setdefault('retrieval', {}).setdefault(metric, []).append(value)
        sources[name] = source

    if not grouped:
        raise FileNotFoundError(f'{dataset} 下没有可汇总的结果文件（metrics_*.json / recall_result.*）')

    payload = {'dataset': dataset, 'confidence': confidence,
               'run_count': len(used), 'runs': used, 'skipped_runs': skipped,
               'retrieval_sources': sources, 'groups': {}}

    for tag, metrics in sorted(grouped.items()):
        payload['groups'][tag] = print_block(f'{dataset}  {tag}', metrics, confidence)

    if skipped:
        print(f'\n跳过（无结果文件）: {skipped}')
    print('\n数据来源:')
    for name in used:
        print(f'  {name[:8]}  recall={sources.get(name)}')

    out = os.path.join(CURRENT_DIR, 'runs', dataset, 'results', 'aggregate_eval.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)
    print(f'\nsaved: {out}')
    return payload


def aggregate_all(confidence=0.95):
    """遍历 runs/<dataset>/，各自汇总；没有结果文件的数据集只提示不报错。"""
    results_dirs = sorted(glob.glob(os.path.join(CURRENT_DIR, 'runs', '*', 'results')))
    if not results_dirs:
        print('no LLaRA_old/runs/<dataset>/results found')
        return {}
    all_summary = {}
    for results_dir in results_dirs:
        dataset = os.path.basename(os.path.dirname(results_dir))
        try:
            all_summary[dataset] = aggregate(dataset, confidence)
        except FileNotFoundError as exc:
            print(f'\n######## dataset: {dataset} 跳过 ########')
            print(f'  {exc}')
    return all_summary


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        aggregate(sys.argv[1])
    else:
        aggregate_all()
