"""验证 LLaRA_old 评测链路的数据：两套候选集的行序 + CF 物品向量。
"""

import argparse
import csv
import glob
import json
import os
import pickle
import sys

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

from config import Config


def pick(pattern: str, run: str, what: str) -> str:
    """按 glob 找文件；run 是 run_id 子串，多份时取字典序第一。"""
    hits = sorted(glob.glob(pattern))
    if run:
        hits = [h for h in hits if run in os.path.basename(os.path.dirname(h))]
    if not hits:
        sys.exit(f'没找到{what}，pattern: {pattern}')
    if len(hits) > 1:
        print(f'  note: {what} 有 {len(hits)} 份，取第一份（可用 --run 指定）')
    return hits[0]


def count_concepts(dataset_dir: str) -> int:
    path = os.path.join(PROJECT_ROOT, dataset_dir, 'need_data', 'entities', 'concepts.csv')
    with open(path, encoding='utf-8') as f:
        return sum(1 for _ in csv.reader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='mid_data')
    ap.add_argument('--run', default='', help='run_id 的子串（同时作用于候选集与特征）')
    args = ap.parse_args()

    config = Config(dataset_dir=f'data/MOOCCube/{args.dataset}')
    rows_path = os.path.join(PROJECT_ROOT, 'prepare_data', 'mooc_sasrec_augllm2',
                             'test_concept_data.p')
    rows = pickle.load(open(rows_path, 'rb'))
    print(f'dataset : {args.dataset}（{len(rows)} 行 test）')
    ok = True

    # ---- A. 第一阶段掩码 top20 ----
    print('\n[A] top20_best_align_test_masked.pkl  ↔  test_concept_data.p')
    pkl_a = pick(os.path.join(PROJECT_ROOT, 'ALLMRec_old', 'runs', args.dataset, 'results', '*',
                              'top20_best_align_test_masked.pkl'), args.run, ' align 候选集')
    print(f'  {pkl_a}')
    a = pickle.load(open(pkl_a, 'rb'))
    topk, truth = a['topk_items'], a['truth_labels']
    n_ok = len(topk) == len(truth) == len(rows)
    bad = sum(1 for i in range(len(rows)) if int(truth[i]) + 1 != int(rows[i][2])) if n_ok else -1
    width = sorted({len(x) for x in topk})
    dup = sum(1 for x in topk if len(set(x)) != len(x))
    print(f'  行数 {len(topk)}/{len(truth)}/{len(rows)}  label 不一致 {bad}  '
          f'行宽 {width}  重复 {dup}'
          f'  {"OK" if n_ok and bad == 0 and width == [20] and dup == 0 else "FAIL"}')
    ok &= n_ok and bad == 0 and width == [20] and dup == 0

    hit = sum(1 for i in range(len(topk)) if truth[i] in topk[i]) / max(len(topk), 1)
    jpath = os.path.join(os.path.dirname(pkl_a), 'eval_best_align.json')
    if os.path.exists(jpath):
        hr = json.load(open(jpath))['test']['hr@20']
        print(f'  正例在候选里 {hit:.5f}  vs  eval json hr@20 {hr:.5f}'
              f'  {"一致" if abs(hr - hit) < 1e-9 else "不一致  FAIL"}')
        ok &= abs(hr - hit) < 1e-9

    # ---- B. sasrec_old 全局 top20 ----
    print('\n[B] top20_best_sasrec_old_test_global.pkl  ↔  test_concept_data.p')
    pkl_b = pick(os.path.join(PROJECT_ROOT, 'sasrec_old', 'runs', args.dataset, 'results', '*',
                              'top20_best_sasrec_old_test_global.pkl'), args.run, ' global 候选集')
    print(f'  {pkl_b}')
    b = pickle.load(open(pkl_b, 'rb'))
    gtopk, gtru = b['topk_items'], b['truth_by_user']
    want = {}
    for r in rows:
        want.setdefault(int(r[0]), set()).add(int(r[2]) - 1)
    bad_u = [u for u in range(len(gtopk)) if {int(v) for v in gtru[u]} != want.get(u, set())]
    gwidth = sorted({len(x) for x in gtopk})
    gdup = sum(1 for x in gtopk if len(set(x)) != len(x))
    print(f'  用户数 {len(gtopk)}/{len(gtru)}  truth 集合不一致 {len(bad_u)}  '
          f'行宽 {gwidth}  重复 {gdup}'
          f'  {"OK" if len(gtopk) == len(gtru) and not bad_u and gwidth == [20] and gdup == 0 else "FAIL"}')
    ok &= len(gtopk) == len(gtru) and not bad_u and gwidth == [20] and gdup == 0

    hits = sum(len({int(c) for c in gtopk[u]} & want.get(u, set())) for u in range(len(gtopk)))
    prec = hits / (20 * max(len(gtopk), 1))
    jpath = os.path.join(os.path.dirname(pkl_b), 'eval_best_sasrec_old.json')
    if os.path.exists(jpath):
        ref = json.load(open(jpath))['test_global']['precision@20']
        print(f'  命中合计 {hits}（人均 {hits / len(gtopk):.2f}）→ precision@20 {prec:.5f}'
              f'  vs  eval json {ref:.5f}  {"一致" if abs(ref - prec) < 1e-9 else "不一致  FAIL"}')
        ok &= abs(ref - prec) < 1e-9

    # ---- C. CF 物品向量 ----
    print('\n[C] Emb_k.npy（e_i，投影层输入）')
    k_path = pick(os.path.join(PROJECT_ROOT, 'sasrec_old', 'runs', args.dataset, 'prepare_data',
                               '*', 'Emb_k.npy'), args.run, ' Emb_k')
    k_emb = np.load(k_path)
    n_concept = count_concepts(config.dataset_dir)
    print(f'  {k_path}')
    print(f'  shape {k_emb.shape}  期望 (行={n_concept}, 列={config.embed_size})'
          f'  {"OK" if k_emb.shape == (n_concept, config.embed_size) else "FAIL"}')
    ok &= k_emb.shape == (n_concept, config.embed_size)

    print('\n' + ('PASS' if ok else 'FAIL'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
