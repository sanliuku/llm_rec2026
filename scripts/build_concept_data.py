"""按 sasrec_old 的方式把 user-video 序列平铺成 user-concept 序列，生成 train/val/test。

每行: [user_index, his_concept_batch, target_concept]

- his_concept_batch: 目标 concept 之前的历史 concept，取最近 20 个，左侧 padding 0 到定长 20
- train: 平铺后逐个 concept 预测下一个（不再按 video 边界构造历史）
- val/test: target = 该 video 中从未在历史出现过的 concept
- concept id 与 sasrec_old 一致整体 +1，0 留给 padding

数据源: data/MOOCCube/mid_data
输出:   prepare_data/mooc_sasrec_augllm2/{train,val,test}_concept_data.p

用法:
    python scripts/build_concept_data.py
    python scripts/build_concept_data.py --limit 5   # 只跑前 5 个用户做冒烟
"""

import argparse
import os
import pickle as pkl

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'MOOCCube', 'small_data')
NEED_DIR = os.path.join(DATA_DIR, 'need_data')
OUT_DIR = os.path.join(PROJECT_ROOT, 'prepare_data', 'mooc_sasrec_augllm')

MAX_HIS = 20  # 历史 concept 长度


def load_dicts():
    columns = ['index', 'id']
    user_df = pd.read_csv(os.path.join(NEED_DIR, 'entities', 'users.csv'), header=None, names=columns)
    video_df = pd.read_csv(os.path.join(NEED_DIR, 'entities', 'videos.csv'), header=None, names=columns)
    index2user = dict(zip(user_df['index'], user_df['id']))
    video2index = dict(zip(video_df['id'], video_df['index']))

    with open(os.path.join(NEED_DIR, 'relations', 'vk.p'), 'rb') as f:
        vk_raw = pkl.load(f)
    # concept id 整体 +1，0 留给 padding
    vk_dict = {vid: [c + 1 for c in concepts] for vid, concepts in vk_raw.items()}
    return index2user, video2index, vk_dict


def flatten_videos(videos, vk_dict):
    """把 video 序列按 vk 关系平铺成 concept 序列（与 sasrec_old.flatten_concepts 一致）。"""
    concepts = []
    for vid in videos:
        concepts.extend(vk_dict.get(vid, []))
    return concepts


def pad_his(concepts, max_len=MAX_HIS):
    """取最后 max_len 个 concept，左侧 padding 0 到定长。"""
    seq = list(concepts[-max_len:])
    return [0] * (max_len - len(seq)) + seq


def user_videos(df, uid, video2index):
    return [video2index[v] for v in df[df['id'] == uid]['vid'].tolist()]


def build_train(index2user, video2index, vk_dict, df, limit=None):
    """训练：把 video 序列平铺成 concept 序列，按 21 的定长窗口（20 历史 + 1 目标）滑动。

    - 序列长度 <= 21：只生成 1 行，历史 = 前面全部 concept（左 padding 到 20），目标 = 最后一个 concept
    - 序列长度 > 21：k 从 20 到末尾，历史 = concepts[k-20:k]，目标 = concepts[k]
    """
    train_data = []
    for index in range(len(index2user)):
        if limit is not None and index >= limit:
            break
        videos = user_videos(df, index2user[index], video2index)
        if not videos:
            continue
        concepts = flatten_videos(videos, vk_dict)
        
        batch_size = 21
        if len(concepts) <= batch_size:
            train_data.append([index, pad_his(concepts[:-1]), concepts[-1]])
            continue

        for k in range(batch_size-1, len(concepts)):
            train_data.append([index, concepts[:k], concepts[k]])
    return train_data


def build_val(index2user, video2index, vk_dict, train_val_df, limit=None):
    """train_val 的最后一个 video 作 target，历史 = 之前全部 concept 的最后 20 个。"""
    val_data = []
    empty = 0
    for index in range(len(index2user)):
        if limit is not None and index >= limit:
            break
        videos = user_videos(train_val_df, index2user[index], video2index)
        if not videos:
            continue
        target_video = videos[-1]
        his_concepts = flatten_videos(videos[:-1], vk_dict)
        # 排除集合用完整历史，避免 target 与历史 concept 重合
        targets = sorted(set(vk_dict.get(target_video, [])) - set(his_concepts))
        if not targets:
            empty += 1
            continue
        his = pad_his(his_concepts)
        for concept in targets:
            val_data.append([index, his, concept])
    if empty:
        print(f'val: {empty} users have empty target, skipped')
    return val_data


def build_test(index2user, video2index, vk_dict, train_val_df, total_df, limit=None):
    """train_val 全部 video 作历史（最后 20 个 concept），target = 测试期 video 的 concept 去掉历史。"""
    test_data = []
    empty = 0
    for index in range(len(index2user)):
        if limit is not None and index >= limit:
            break
        uid = index2user[index]
        tv_videos = user_videos(train_val_df, uid, video2index)
        if not tv_videos:
            continue
        all_videos = user_videos(total_df, uid, video2index)
        test_videos = set(all_videos) - set(tv_videos)

        his_concepts = flatten_videos(tv_videos, vk_dict)
        targets = sorted(set(flatten_videos(test_videos, vk_dict)) - set(his_concepts))
        if not targets:
            empty += 1
            continue
        his = pad_his(his_concepts)
        for concept in targets:
            test_data.append([index, his, concept])
    if empty:
        print(f'test: {empty} users have empty target, skipped')
    return test_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='只处理前 N 个用户（调试用）')
    args = parser.parse_args()

    index2user, video2index, vk_dict = load_dicts()
    with open(os.path.join(DATA_DIR, 'train_df2.pkl'), 'rb') as f:
        train_df = pkl.load(f)
    with open(os.path.join(DATA_DIR, 'train_val_df2.pkl'), 'rb') as f:
        train_val_df = pkl.load(f)
    with open(os.path.join(DATA_DIR, 'filtered_second_df2.pkl'), 'rb') as f:
        total_df = pkl.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    splits = {
        'train': build_train(index2user, video2index, vk_dict, train_df, args.limit),
        'val': build_val(index2user, video2index, vk_dict, train_val_df, args.limit),
        'test': build_test(index2user, video2index, vk_dict, train_val_df, total_df, args.limit),
    }
    for name, rows in splits.items():
        path = os.path.join(OUT_DIR, f'{name}_concept_data.p')
        with open(path, 'wb') as f:
            pkl.dump(rows, f, protocol=pkl.HIGHEST_PROTOCOL)
        print(f'{name}: {len(rows)} rows -> {path}')


if __name__ == '__main__':
    main()
