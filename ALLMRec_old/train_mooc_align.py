"""A-LLMRec 第一阶段的训练脚本（config 驱动）：冻结 CF 模型，只训对齐网络。
"""

import json
import os
import pickle as pkl
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

from align_data import AlignTrainDataset, get_log_seqs, load_windows
from align_utils import (cf_baseline, evaluate, load_cf_config, load_cf_model,
                         load_text_model)
from config_align import AlignConfig
from mooc_align import MoocAlign
from sasrec_old.config import Config as SasrecConfig


class _Tee:
    """把 stdout 同时写到终端和 run 目录下的 train.log。"""

    def __init__(self, stream, file):
        self._stream = stream
        self._file = file

    def write(self, msg):
        self._stream.write(msg)
        self._file.write(msg)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random
    random.seed(seed)
    np.random.seed(seed)


def build_optimizer(model, config):
    """对齐 MLP 与 SBERT 分两组参数，各用自己的 lr；sbert_lr=None 时退化成源码的单 lr。"""
    mlp_params, sbert_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (sbert_params if name.startswith('text_model.') else mlp_params).append(param)

    groups = [{'params': mlp_params, 'lr': config.lr, 'tag': 'align_mlp'}]
    if sbert_params:
        sbert_lr = config.sbert_lr if config.sbert_lr is not None else config.lr
        groups.append({'params': sbert_params, 'lr': sbert_lr, 'tag': 'sbert'})

    for g in groups:
        print(f'trainable [{g["tag"]}]: {sum(p.numel() for p in g["params"]):,} params, lr={g["lr"]}')
    print(f'trainable total: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} / '
          f'{sum(p.numel() for p in model.parameters()):,} (all params)')
    return torch.optim.Adam(groups, lr=config.lr, betas=(0.9, 0.98))


def load_data(config: AlignConfig, device):
    """构造 stage-1 的全部数据：CF 窗口（train）+ 每用户 val/test 序列 + 负例表 + 掩码。

    数据口径一律以冻结 CF 那个 run 的 config.json 为准，不从 AlignConfig 另设。
    """
    cf_json = load_cf_config(config)
    cf_model, cf_config = load_cf_model(config, device)

    if os.path.normpath(cf_json.get('dataset_dir', '')) != os.path.normpath(config.dataset_dir):
        raise ValueError(f'dataset_dir={config.dataset_dir} 与 CF 的 {cf_json["dataset_dir"]} 不一致')

    cf_cfg = SasrecConfig.from_json(os.path.join(config.cf_run_dir(PROJECT_ROOT), 'config.json'))
    windows = load_windows(cf_cfg)
    text_model = load_text_model(config, cf_config.concept_num, device)

    need_dir = config.need_data_dir(PROJECT_ROOT)
    adj = np.load(os.path.join(need_dir, 'matrix/adj.npy')) > 0   # val 掩码：train+val
    ua = np.load(os.path.join(need_dir, 'matrix/ua.npy')) > 0     # test 掩码：全交互
    negative = pkl.load(open(os.path.join(need_dir, 'negatives/negative.p'), 'rb'))
    test_negative = pkl.load(open(os.path.join(need_dir, 'negatives/test_negative.p'), 'rb'))
    num_user = adj.shape[0]

    val_seqs = get_log_seqs(windows['val_seq_dict'], num_user, cf_config.max_len)
    test_seqs = get_log_seqs(windows['test_seq_dict'], num_user, cf_config.max_len)

    dataset = AlignTrainDataset(windows, cf_config.concept_num)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    print(f'train  : {len(dataset)} 窗口 <- CF 滑动窗口 (max_len={cf_config.max_len}, '
          f'stride={cf_cfg.window_stride})')
    print(f'val    : {num_user} 用户 × {cf_config.max_len} 长度序列，'
          f'{sum(len(v) for v in negative.values())} 条 query，掩码 adj.npy')
    print(f'test   : {num_user} 用户 × {cf_config.max_len} 长度序列，'
          f'{sum(len(v) for v in test_negative.values())} 条 query，掩码 ua.npy')
    print(f'users: {num_user}, concepts: {cf_config.concept_num}')

    print('冻结 CF 基线（原始 E_i，不经过对齐 MLP）：')
    base_val = cf_baseline(cf_model, val_seqs, adj, negative, device)
    base_test = cf_baseline(cf_model, test_seqs, ua, test_negative, device)

    return {
        'cf_model': cf_model,
        'cf_config': cf_config,
        'text_model': text_model,
        'adj': adj,
        'ua': ua,
        'negative': negative,
        'test_negative': test_negative,
        'val_seqs': val_seqs,
        'test_seqs': test_seqs,
        'train_loader': train_loader,
        'baseline': {'val': base_val, 'test': base_test},
    }


def train(model, dataloader, optimizer, device):
    model.train()
    keys = ('total', 'rec', 'matching', 'item_recon', 'text_recon')
    dkeys = ('code_std', 'cos_ee', 'recon_rel', 'logit_pos', 'logit_neg')
    total = {k: 0.0 for k in keys}
    diag = {k: 0.0 for k in dkeys}
    # 末位 target 恒非 0，但窗口里的 padding 位置会带出 0 目标，这里保持分母一致
    for seq, pos, neg in tqdm(dataloader, desc='train', leave=False, ncols=0):
        seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)
        optimizer.zero_grad()
        out = model(seq, pos, neg)
        out['total'].backward()
        optimizer.step()
        for k in keys:
            total[k] += out[k].item() * seq.size(0)
        for k in dkeys:
            diag[k] += out['diag'][k] * seq.size(0)
    n = len(dataloader.dataset)
    return {**{k: v / n for k, v in total.items()}, **{k: v / n for k, v in diag.items()}}


def main(config: AlignConfig):
    seed_everything(config.seed)
    device = torch.device(config.device)
    run_dir = config.create_run_dir(PROJECT_ROOT)
    log_file = open(os.path.join(run_dir, 'train.log'), 'a', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, log_file)

    try:
        print(f'run dir : {run_dir}')
        data = load_data(config, device)

        model = MoocAlign(data['cf_model'], data['text_model'], config).to(device)
        optimizer = build_optimizer(model, config)
        optimizer.zero_grad()

        max_val = 0.0
        test_mrr = float('nan')
        test_hr20 = float('nan')
        init_index = 0
        metrics_path = os.path.join(run_dir, 'metrics.jsonl')
        for epoch in range(config.epochs):
            train_parts = train(model, data['train_loader'], optimizer, device)
            val_metrics = evaluate(model, data['val_seqs'], data['adj'],
                                   data['negative'], device)
            val_mrr = val_metrics['mrr']
            if epoch % config.eval_interval == 0:
                test_metrics = evaluate(model, data['test_seqs'], data['ua'],
                                        data['test_negative'], device)
                test_mrr = test_metrics['mrr']
                test_hr20 = test_metrics['hr@20']

            if val_mrr > max_val:
                torch.save(model.state_dict(), os.path.join(run_dir, 'best_align.pth'))
                max_val = val_mrr
                init_index = 0
            else:
                init_index += 1

            base_val = data['baseline']['val']['mrr']
            base_test = data['baseline']['test']['mrr']
            print(f'epoch: {epoch}, train_loss: {train_parts["total"]:.4f}, '
                  f'rec: {train_parts["rec"]:.4f}, matching: {train_parts["matching"]:.4f}, '
                  f'item_recon: {train_parts["item_recon"]:.4f}, text_recon: {train_parts["text_recon"]:.4f}, '
                  f'val_mrr: {val_mrr:.5f} (基线 {base_val:.5f}, {(val_mrr / base_val - 1) * 100:+.1f}%), '
                  f'test_mrr: {test_mrr:.5f} (基线 {base_test:.5f}, '
                  f'{(test_mrr / base_test - 1) * 100:+.1f}%) | '
                  f'code_std: {train_parts["code_std"]:.4f}, cos_ee: {train_parts["cos_ee"]:.4f}, '
                  f'recon_rel: {train_parts["recon_rel"]:.4f}, '
                  f'logit: {train_parts["logit_pos"]:+.2f}/{train_parts["logit_neg"]:+.2f}')
            with open(metrics_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'epoch': epoch,
                    'train_loss': train_parts['total'],
                    'train_rec': train_parts['rec'],
                    'train_matching': train_parts['matching'],
                    'train_item_recon': train_parts['item_recon'],
                    'train_text_recon': train_parts['text_recon'],
                    'code_std': train_parts['code_std'],
                    'cos_ee': train_parts['cos_ee'],
                    'recon_rel': train_parts['recon_rel'],
                    'logit_pos': train_parts['logit_pos'],
                    'logit_neg': train_parts['logit_neg'],
                    'val_mrr': val_mrr,
                    'val_hr@20': val_metrics['hr@20'],
                    'test_mrr': test_mrr,
                    'test_hr@20': test_hr20,
                    'baseline_val_mrr': base_val,
                    'baseline_test_mrr': base_test,
                }, ensure_ascii=False) + '\n')

            if init_index > config.early_stop:
                print(f'early stop at epoch {epoch}')
                break

        print('train done')
    finally:
        sys.stdout = original_stdout
        log_file.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='A-LLMRec 第一阶段：文本侧在线/离线切换')
    side = parser.add_mutually_exclusive_group()
    side.add_argument('--online', action='store_true',
                      help='文本侧在线跑 SBERT（覆盖 config.text_emb_path 为 None）')
    side.add_argument('--offline', nargs='?', const='', default=None, metavar='NPY',
                      help='文本侧用离线嵌入表；不跟路径则沿用 config.text_emb_path')
    parser.add_argument('--seed', type=int, default=None, help='只跑一个 seed（默认 5 个）')
    args = parser.parse_args()

    config = AlignConfig()
    if args.online:
        config.text_emb_path = None
    elif args.offline is not None: 
        config.text_emb_path = args.offline or config.text_emb_path
    print(f'text side: {config.text_emb_path or "online sbert"}')

    seeds = [args.seed] if args.seed is not None else [12, 22, 32, 42, 52]
    for seed in seeds:
        config.seed = seed
        config.exist = True
        main(config)
