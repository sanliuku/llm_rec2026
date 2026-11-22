"""评测候选集的统一加载：把两种 pkl 归一成「一条样本 → 一组候选 + 一组可接受的答案」。

`ALLMRec_llm/infer_mooc_llm.py` 和 `LLaRA_old/infer_mooc_llara_old.py` 都从这里取样本，
保证两条链路吃的是同一份数据。

id 一律转成 **1-based**（Emb_k 行号 / id2title 的键都是这个坐标），pkl 里存的是 0-based。

  align_masked   ALLMRec_old 第一阶段 per-query 掩码 top20，**per-row**
                 <run>/top20_best_align_test_masked.pkl
                 {'topk_items': N×20, 'truth_labels': N}，行序 == test_concept_data.p
                 答案 = 该行唯一 target，行数 = 53911
  sasrec_global  sasrec_old 的 per-user 全库排序，**per-user**
                 <run>/top20_best_sasrec_old_test_global.pkl
                 {'topk_items': 用户数×20, 'truth_by_user': {uid: [该用户全部 test 正例]}}
                 答案 = truth_by_user 那 18 个，行数 = 1000

global 是「不看时序、只把用户没交互过的排一遍」，一个用户一个 query，套 per-row 会把
同一份候选重复算 18 遍。所以 per-user 下：
  candidate = 该用户 global top20
  history   = 该用户第一条样本的历史（= test 之前那段，截断到 num_his）
  row       = 该用户第一条样本在 concept_data 里的行号（x^u 按这个取）
  answer    = truth_by_user[uid]（命中任意一个就算对）

两个 topk_items 都只在「正例本身排进前 20」时才含正例（align 8.2%、global 5.0%）。
候选原样交给 LLM，不做任何补正例：没被第一阶段召回就是没召回，那部分样本 rank=-1，
如实计入——`hr@1` 才是端到端（第一阶段召回 × LLM 排序）的数字。
"""

import glob
import os
import pickle as pkl
import random

CAND_SOURCES = ('align_masked', 'sasrec_global')
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

_AUTO_PATTERN = {
    'align_masked': ('ALLMRec_old', 'runs', '{dataset}', 'results', '*',
                     'top20_best_align_{split}_masked.pkl'),
    'sasrec_global': ('sasrec_old', 'runs', '{dataset}', 'results', '*',
                      'top20_best_sasrec_old_{split}_global.pkl'),
}


def resolve_cand_path(config, source: str, split: str, explicit: str = '') -> str:
    """定位候选集 pkl：explicit 优先，否则按 source 在对应 runs 目录下自动找。

    同一个 dataset 下有多份（每个 seed 一套）时取字典序第一个，并打印总数提醒。
    sasrec_global 只有 test 口径，split=val 时会直接报找不到。
    """
    if source not in CAND_SOURCES:
        raise ValueError(f'cand_source 只能是 {CAND_SOURCES}，收到 {source!r}')
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f'候选集不存在: {explicit}')
        return explicit

    pattern = os.path.join(PROJECT_ROOT, *_AUTO_PATTERN[source]).format(
        dataset=config.dataset_name, split=split)
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(
            f'没找到 {source} 的候选集，pattern: {pattern}\n'
            f'  先用 --cand_path 指定，或先跑对应的第一阶段评测')
    if len(hits) > 1:
        print(f'[cand] {source} 有 {len(hits)} 份，取第一份: {os.path.basename(os.path.dirname(hits[0]))}')
    return hits[0]


def _his(row, num_his: int) -> list:
    return [int(x) for x in row[1] if x][-num_his:]


def _check_rows(path: str, targets, rows) -> None:
    """候选集自带的 label 必须和 concept_data 的 target 逐行一致，否则行序错了。"""
    if len(targets) != len(rows):
        raise ValueError(f'{path} 有 {len(targets)} 行，concept_data {len(rows)} 行，对不上')
    bad = [i for i in range(len(rows)) if int(rows[i][2]) != targets[i]]
    if bad:
        raise ValueError(f'{path} 有 {len(bad)} 行 label 与 target 不一致（首例 row={bad[0]}: '
                         f'label={targets[bad[0]]} target={rows[bad[0]][2]}）')


def load_source(source: str, path: str, rows, num_his: int) -> tuple:
    """读候选集，返回 (items, mode)。

    items : [{'row', 'uid', 'his', 'cands', 'answer'}]，cands/answer 都是 1-based
    mode  : 'per-row'（align）或 'per-user'（global）
    """
    data = pkl.load(open(path, 'rb'))

    if source == 'align_masked':
        cands = [[int(c) + 1 for c in x] for x in data['topk_items']]
        targets = [int(t) + 1 for t in data['truth_labels']]
        _check_rows(path, targets, rows)
        items = [{'row': i, 'uid': int(r[0]), 'his': _his(r, num_his),
                  'cands': cands[i], 'answer': {targets[i]}} for i, r in enumerate(rows)]
        return items, 'per-row'

    # sasrec_global：候选与真值都按用户存，一个用户一条样本
    first_row, his_by_user = {}, {}
    for i, r in enumerate(rows):
        u = int(r[0])
        if u not in first_row:
            first_row[u] = i
            his_by_user[u] = _his(r, num_his)

    topk, truth_by_user = data['topk_items'], data['truth_by_user']
    if len(topk) != len(truth_by_user):
        raise ValueError(f'{path} 的 topk_items 与 truth_by_user 用户数不一致')
    missing = [u for u in range(len(topk)) if u not in first_row]
    if missing:
        raise ValueError(f'{path} 有 {len(missing)} 个用户在 concept_data 里找不到（首例 uid={missing[0]}）')

    items = [{'row': first_row[u], 'uid': u, 'his': his_by_user[u],
              'cands': [int(c) + 1 for c in topk[u]],
              'answer': {int(v) + 1 for v in truth_by_user[u]}} for u in range(len(topk))]
    return items, 'per-user'


def build_samples(items, seed: int, shuffle_cand: bool = True) -> list:
    """洗候选顺序（同一行固定种子，可复现）。候选原样，不增不删。"""
    samples = []
    for i, it in enumerate(items):
        c = list(it['cands'])
        if shuffle_cand:
            random.Random(seed + i).shuffle(c)
        samples.append({**it, 'cands': c})

    n = len(samples)
    width = sorted({len(s['cands']) for s in samples})
    ans = [len(s['answer']) for s in samples]
    inside = sum(1 for s in samples if s['answer'] & set(s['cands']))
    print(f'[cand] {n} 条，候选 {width} 个/条，答案 {min(ans)}~{max(ans)} 个/条；'
          f'答案落在候选里 {inside}/{n}（{inside / n:.1%}），其余 rank=-1')
    return samples


def load_candidates(config, source: str, split: str, rows, explicit: str = '',
                    num_his: int = 20, seed: int = 12,
                    shuffle_cand: bool = True) -> tuple:
    """一步到位：定位文件 → 读 → 拼样本。返回 (samples, path)。"""
    path = resolve_cand_path(config, source, split, explicit)
    print(f'[cand] source={source}  {path}')
    items, mode = load_source(source, path, rows, num_his)
    print(f'[cand] {mode}')
    return build_samples(items, seed, shuffle_cand), path
