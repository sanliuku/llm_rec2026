"""把每个 stage-1 run 的 checkpoint 导出成 npy，供第二阶段 LLM 使用。
"""

import os
import pickle as pkl
import sys

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

from align_utils import load_cf_model, load_text_model, resolve
from config_align import AlignConfig, iter_run_dirs
from mooc_align import MoocAlign


def split_seqs(p_path: str, max_len: int) -> np.ndarray:
    """按 .p 的行构造模型输入序列，行序与 .p 完全一致（与 sasrec_old/prepare.py 相同）。

    每行取 his_concept_batch（已 +1、已左 padding），再左侧补 0 到 max_len。
    """
    with open(p_path, 'rb') as f:
        rows = pkl.load(f)
    seqs = np.zeros([len(rows), max_len], dtype=np.int32)
    for i, row in enumerate(rows):
        hist = np.asarray(row[1], dtype=np.int32)[-max_len:]
        if hist.size:
            seqs[i, max_len - hist.size:] = hist
    return seqs


def user_features(model, seqs: np.ndarray, device: torch.device, batch_size: int = 1000) -> np.ndarray:
    """逐 batch 跑冻结 CF 的 log2feats，返回 (样本数, emb_dim) 的末位置表征。"""
    feats = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = torch.from_numpy(seqs[i:i + batch_size]).to(device)
            log_feats, _ = model.cf.log2feats(batch)
            feats.append(log_feats[:, -1, :].cpu().numpy())
    return np.concatenate(feats, axis=0)


def prepare_run(config: AlignConfig, run_dir: str, device: torch.device, split_paths: dict):
    """导出单个 run 的 npy；对应 ckpt 不存在则跳过。"""
    ckpt_path = os.path.join(run_dir, 'best_align.pth')
    if not os.path.exists(ckpt_path):
        print(f'[skip] ckpt not found: {ckpt_path}')
        return

    cf_model, cf_config = load_cf_model(config, device)
    text_model = load_text_model(config, cf_config.concept_num, device)
    model = MoocAlign(cf_model, text_model, config).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # 键沿用 .p 的文件名（train / val / test），行序与 .p 一一对应
    seq_dict = {}
    for split, p_path in split_paths.items():
        p_path = resolve(p_path)
        if not os.path.exists(p_path):
            print(f'[skip] {split} p not found: {p_path}')
            continue
        name = os.path.basename(p_path).replace('_concept_data.p', '')
        seq_dict[name] = split_seqs(p_path, cf_config.max_len)
        print(f'{split} -> {name}: {seq_dict[name].shape[0]} 行 <- {p_path}')

    if not seq_dict:
        print(f'[skip] no split p available for {config.dataset_name}')
        return

    out_dir = os.path.join(config.prepare_data_dir(PROJECT_ROOT), config.run_id)
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        # 下游 LLM 只消费对齐后的 concept 向量 e_i（与 prepare.py 的 Emb_k 同形），
        # 去掉 padding 行，与 concepts.csv 的行对齐
        Emb_k = model.joint_item_embeddings().cpu().numpy()[1:]
    np.save(os.path.join(out_dir, 'Emb_k.npy'), Emb_k)
    print(f'Emb_k {Emb_k.shape}')

    for name, seqs in seq_dict.items():
        feats = user_features(model, seqs, device)
        np.save(os.path.join(out_dir, f'{name}_user_features.npy'), feats)
        print(f'{name}_user_features.npy: {feats.shape}')

    print(f'[{config.dataset_name}] {config.run_id}: -> {out_dir}')


def main(split_paths: dict, skip=()):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    run_dirs = iter_run_dirs(PROJECT_ROOT)  # runs 下所有实验
    print(f'run examples: {len(run_dirs)}')

    for run_dir in run_dirs:
        if os.path.basename(os.path.normpath(run_dir)) in skip:
            print(f'[skip] {run_dir}')
            continue
        config = AlignConfig.load(run_dir, PROJECT_ROOT)
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
