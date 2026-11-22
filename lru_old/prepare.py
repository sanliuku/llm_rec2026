"""把 lru_old 每个 run 的 checkpoint 导出成 npy（concept 表 + 用户表征）。
"""

import os
import pickle as pkl
import sys
from types import SimpleNamespace

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

from config import Config, iter_run_dirs
from mooc_lru_old import LRUOld


def count_rows(path: str) -> int:
    """统计 csv 行数（这些 csv 没有表头）。"""
    with open(path, encoding='utf-8') as f:
        return sum(1 for _ in f)


def split_seqs(p_path: str, max_len: int) -> np.ndarray:
    """按 .p 的行构造模型输入序列，行序与 .p 完全一致。

    每行取 his_concept_batch（已 +1、已左 padding），再左侧补 0 到 max_len。
    """
    with open(p_path, 'rb') as f:
        rows = pkl.load(f)
    seqs = np.zeros([len(rows), max_len], dtype=np.int32)
    for i, row in enumerate(rows):
        hist = row[1][-max_len:]
        if hist:
            seqs[i, max_len - len(hist):] = hist
    return seqs


def user_features(model, log_seqs: np.ndarray, device: torch.device,
                  batch_size: int = 1000) -> np.ndarray:
    """逐 batch 跑 log2feats，返回 (样本数, emb_dim) 的末位置表征。"""
    feats = []
    with torch.no_grad():
        for i in range(0, len(log_seqs), batch_size):
            batch = torch.from_numpy(log_seqs[i:i + batch_size]).to(device)
            log_feats, _ = model.log2feats(batch)
            feats.append(log_feats[:, -1, :].cpu().numpy())
    return np.concatenate(feats, axis=0)


def resolve_p(p_path: str) -> str:
    """把 __main__ 里的相对路径（相对项目根目录）解析成绝对路径。"""
    return p_path if os.path.isabs(p_path) else os.path.normpath(os.path.join(PROJECT_ROOT, p_path))


def prepare_run(config: Config, run_dir: str, device: torch.device, split_paths: dict):
    """导出单个 run 的 npy；对应 ckpt 不存在则跳过。"""
    path = os.path.join(PROJECT_ROOT, config.dataset_dir, 'need_data') + '/'
    model_config = SimpleNamespace(
        concept_num=count_rows(path + 'entities/concepts.csv'),
        emb_dim=config.emb_dim,
        max_len=config.max_len,
        dropout_rate=config.dropout_rate,
        num_blocks=config.num_blocks,
        lru_r_min=config.lru_r_min,
        lru_r_max=config.lru_r_max,
        device=device,
    )
    model = LRUOld(model_config).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    ckpt_path = os.path.join(run_dir, 'best_lru_old.pth')
    if not os.path.exists(ckpt_path):
        print(f'[skip] ckpt not found: {ckpt_path}')
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # 键沿用 .p 的文件名（train / val / test），行序与 .p 一一对应
    seq_dict = {}
    for split, p_path in split_paths.items():
        p_path = resolve_p(p_path)
        if not os.path.exists(p_path):
            print(f'[skip] {split} p not found: {p_path}')
            continue
        name = os.path.basename(p_path).replace('_concept_data.p', '')
        seq_dict[name] = split_seqs(p_path, config.max_len)
        print(f'{split} -> {name}: {seq_dict[name].shape[0]} 行 <- {p_path}')

    if not seq_dict:
        print(f'[skip] no split p available for {config.dataset_name}')
        return

    out_dir = os.path.join(config.prepare_data_dir(PROJECT_ROOT), config.run_id)
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        Emb_k = model.get_item_embeddings().cpu().numpy()[1:]  # 去掉 padding 行，与 concepts.csv 对齐
    np.save(os.path.join(out_dir, 'Emb_k.npy'), Emb_k)

    for name, seqs in seq_dict.items():
        feats = user_features(model, seqs, device)
        np.save(os.path.join(out_dir, f'{name}_user_features.npy'), feats)
        print(f'{name}_user_features.npy: {feats.shape}')

    print(f'[{config.dataset_name}] {config.run_id}: Emb_k {Emb_k.shape} -> {out_dir}')


def main(split_paths: dict, skip=()):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    run_dirs = iter_run_dirs(PROJECT_ROOT)  # lru_old/runs 下所有实验
    print(f'run examples: {len(run_dirs)}')

    for run_dir in run_dirs:
        if os.path.basename(os.path.normpath(run_dir)) in skip:
            print(f'[skip] {run_dir}')
            continue
        config = Config.load(run_dir, PROJECT_ROOT)
        paths = split_paths.get(config.dataset_name)
        if not paths:
            print(f'[skip] no split p mapping for dataset: {config.dataset_name}')
            continue
        prepare_run(config, run_dir, device, paths)
    print('done')


if __name__ == '__main__':

    # 各数据集划分对应的 train/val/test .p（相对项目根目录）
    p = {
        "mid_data": {
            "train": "prepare_data/mooc_sasrec_augllm2/train_concept_data.p",
            "val": "prepare_data/mooc_sasrec_augllm2/val_concept_data.p",
            "test": "prepare_data/mooc_sasrec_augllm2/test_concept_data.p",
        },
        "small_data": {
            "train": "prepare_data/mooc_sasrec_augllm/train_concept_data.p",
            "val": "prepare_data/mooc_sasrec_augllm/val_concept_data.p",
            "test": "prepare_data/mooc_sasrec_augllm/test_concept_data.p",
        },
    }

    main(p)
