"""把每个 run 的 checkpoint 导出成 npy（Emb_v / Emb_k / 用户表征）。

- run 列表：`runs/<dataset>/results/*` 下所有含 config.json 的 run 目录（iter_run_dirs）。
- 用户表征的数据来源：各数据集划分对应的 .p（见 __main__ 里的 p 映射，按 dataset_name 取）。
  .p 的每一行是 `[user_index, his_videos(0-based), label_concept, label_video]`，
  逐行取 his_videos 的末尾 max_len 个 video、下标 +1（0 留作 padding）左 padding 成模型输入。
- 每个 .p 行产出一个表征（最后一个位置的隐状态），与 .p 行严格一一对应，下游可按行取用。
- npy 输出：`runs/<dataset>/prepare_data/<run_id>/`，每个 run 一个目录，避免多个 run 互相覆盖。
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
from mooc_sasrec import MOOCSASRec


def build_edge_index(path: str, device: str) -> torch.Tensor:
    """vk 关系展开成「video + concept」同构图，供 GAT 使用。"""
    vk_matrix = torch.from_numpy(np.load(path + 'matrix/vk.npy')).float()
    O1 = torch.zeros((vk_matrix.size(0), vk_matrix.size(0)))
    hi = torch.cat((O1, vk_matrix), dim=1)
    O2 = torch.zeros((vk_matrix.size(1), vk_matrix.size(1)))
    lo = torch.cat((vk_matrix.T, O2), dim=1)
    L_vk = torch.cat((hi, lo), dim=0).float().to_sparse_coo()
    return L_vk.indices().to(device)


def count_rows(path: str) -> int:
    """统计 csv 行数（这些 csv 没有表头）。"""
    with open(path, encoding='utf-8') as f:
        return sum(1 for _ in f)


def split_seqs(p_path: str, max_len: int) -> np.ndarray:
    """按 .p 的行构造模型输入序列。

    每行一条序列：取该行 his_videos 的末尾 max_len 个，video 下标 +1（0 为 padding），
    左 padding 到 max_len。行序与 .p 完全一致。
    """
    with open(p_path, 'rb') as f:
        rows = pkl.load(f)
    seqs = np.zeros([len(rows), max_len], dtype=np.int32)
    for i, row in enumerate(rows):
        hist = row[1][-max_len:]
        if hist:
            seqs[i, max_len - len(hist):] = [vid + 1 for vid in hist]
    return seqs


def user_features(model, log_seqs: np.ndarray, device: torch.device,
                  batch_size: int = 1000) -> np.ndarray:
    """逐 batch 跑 log2feats，返回 (样本数, emb_dim) 的末位置表征。"""
    feats = []
    with torch.no_grad():
        for i in range(0, len(log_seqs), batch_size):
            batch = torch.from_numpy(log_seqs[i:i + batch_size]).to(device)
            _, _, log_feats = model.log2feats(batch)
            feats.append(log_feats[:, -1, :].cpu().numpy())
    return np.concatenate(feats, axis=0)


def resolve_p(p_path: str) -> str:
    """把 __main__ 里的相对路径（相对项目根目录）解析成绝对路径。"""
    return p_path if os.path.isabs(p_path) else os.path.normpath(os.path.join(PROJECT_ROOT, p_path))


def prepare_run(config: Config, run_dir: str, device: torch.device, split_paths: dict):
    """导出单个 run 的 npy；对应 ckpt 不存在则跳过。"""
    path = os.path.join(PROJECT_ROOT, config.dataset_dir, 'need_data') + '/'
    model_config = SimpleNamespace(
        user_num=count_rows(path + 'entities/users.csv'),
        video_num=count_rows(path + 'entities/videos.csv'),
        concept_num=count_rows(path + 'entities/concepts.csv'),
        bert_dim=config.bert_dim,
        emb_dim=config.emb_dim,
        max_len=config.max_len,
        dropout_rate=config.dropout_rate,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        device=device,
    )
    model = MOOCSASRec(model_config, edge_index=build_edge_index(path, device),
                       emb_path=path + 'embs/concept_embs_local.npy').to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # 键沿用 .p 的文件名（train / val / eval），行序与 .p 一一对应
    seq_dict = {}
    for split, p_path in split_paths.items():
        p_path = resolve_p(p_path)
        if not os.path.exists(p_path):
            print(f'[skip] {split} p not found: {p_path}')
            continue
        name = os.path.basename(p_path).replace('_data_new.p', '')
        seq_dict[name] = split_seqs(p_path, config.max_len)
        print(f'{split} -> {name}: {seq_dict[name].shape[0]} 行 <- {p_path}')

    if not seq_dict:
        print(f'[skip] no split p available for {config.dataset_name}')
        return

    out_dir = os.path.join(config.prepare_data_dir(PROJECT_ROOT), config.run_id)
    os.makedirs(out_dir, exist_ok=True)
    for alpha in config.alphas:
        ckpt_stem = f'best_sasrec_mooc{int(alpha * 10)}'
        ckpt_path = os.path.join(run_dir, ckpt_stem + '.pth')
        if not os.path.exists(ckpt_path):
            print(f'[skip] ckpt not found: {ckpt_path}')
            continue
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        with torch.no_grad():
            Emb_v, Emb_k = model.get_item_embeddings()
        Emb_v = Emb_v.cpu().numpy()[1:]  # 去掉 padding 行，与 concepts/videos.csv 对齐
        Emb_k = Emb_k.cpu().numpy()[1:]

        np.save(os.path.join(out_dir, 'Emb_v.npy'), Emb_v)
        np.save(os.path.join(out_dir, 'Emb_k.npy'), Emb_k)
        for name, seqs in seq_dict.items():
            feats = user_features(model, seqs, device)
            np.save(os.path.join(out_dir, f'{name}_user_features.npy'), feats)
            print(f'{name}_user_features.npy: {feats.shape}')

        print(f'[{config.dataset_name}] {ckpt_stem}: Emb_v {Emb_v.shape}, Emb_k {Emb_k.shape} '
              f'-> {out_dir}')


def main(split_paths: dict):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    run_dirs = iter_run_dirs(PROJECT_ROOT)  # runs 下所有实验示例
    filtered = ["91118113a3edc436daf089a2e8973917", "19840c33274f44e890a42270b77a0f90"]

    print(f'run examples: {len(run_dirs)}')
    for run_dir in run_dirs:
        if run_dir.split('\\')[-1] in filtered:
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
            "train": "prepare_data/mooc_sasrec_augllm2/train_data_new.p",
            "val": "prepare_data/mooc_sasrec_augllm2/val_data_new.p",
            "test": "prepare_data/mooc_sasrec_augllm2/eval_data_new.p",
        },
        "small_data": {
            "train": "prepare_data/train_data_new.p",
            "val": "prepare_data/val_data_new.p",
            "test": "prepare_data/eval_data_new.p",
        }
    }

    main(p)
