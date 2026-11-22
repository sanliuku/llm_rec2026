"""A-LLMRec 第一阶段（协同嵌入与文本嵌入对齐）的实验配置。
目录结构::

    <save_root>/<dataset>/prepare_data/<run_id>/   # prepare_align.py 导出的联合嵌入
    <save_root>/<dataset>/results/<md5>/           # 一次实验的全部产物
"""

import glob
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field, fields

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class AlignConfig:
    # ---- 数据（继承冻结的 CF run）----
    dataset_dir: str = 'data/MOOCCube/mid_data'
    cf_run: str = 'sasrec_old/runs/mid_data/results'  # results 父目录=按 seed 匹配；具体 run 目录=固定
    cf_ckpt: str = 'best_sasrec_old.pth'

    # ---- 文本侧：text_emb_path 有值走离线表，None 走在线 SBERT ----
    text_emb_path: str | None = 'data/MOOCCube/mid_data/need_data/embs/concept_embs_local.npy'
    concept_csv: str = 'data/MOOCCube/mid_data/need_data/entities/concepts.csv'
    text_pkl: str = 'data/MOOCCube/mid_data/need_data/aug_concepts.pkl'
    text_model: str = 'local_llm/bge-base-zh-v1.5'
    text_dim: int = 768  # 在线 SBERT 的嵌入维度（只在 text_emb_path=None 时校验）
    text_max_length: int = 128
    text_finetune: bool = False  # 源码 self.sbert.train()；False 只用于冻结消融
    sbert_freeze_layers: int = 6  # 只冻底部 k 层（k>0 时必须连 embedding 一起冻）
    freeze_embeddings: bool = True
    sbert_lr: float | None = 3e-5  # None = 与 MLP 共用 lr

    # ---- 模型 ----
    align_dim: int = 128  
    enc_act: str = 'leakyrelu'  

    # ---- loss 权重（式 6）----
    alpha: float = 0.5  # item 重建项
    beta: float = 0.2  # text 重建项

    # ---- 训练 ----
    seed: int = 42
    epochs: int = 15
    early_stop: int = 3
    batch_size: int = 128  
    lr: float = 1e-4
    eval_interval: int = 1
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # ---- 输出 ----
    save_root: str = 'ALLMRec_old/runs'
    exist: bool = field(default=False, metadata={'no_hash': True})  # 是否清空已存在的 run 目录

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self) if not f.metadata.get('no_hash')}

    def tostring(self) -> str:
        """稳定序列化，用于生成 md5。"""
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False,
                          separators=(',', ':'), default=str)

    @property
    def run_id(self) -> str:
        return hashlib.md5(self.tostring().encode('utf-8')).hexdigest()

    @property
    def dataset_name(self) -> str:
        return os.path.basename(os.path.normpath(self.dataset_dir))

    def _base_dir(self, project_root: str = PROJECT_ROOT) -> str:
        base = self.save_root if os.path.isabs(self.save_root) else os.path.join(project_root, self.save_root)
        return os.path.join(base, self.dataset_name)

    def need_data_dir(self, project_root: str = PROJECT_ROOT) -> str:
        """原始实体/关系表目录（entities、matrix）。"""
        return os.path.join(project_root, self.dataset_dir, 'need_data')

    def prepare_data_dir(self, project_root: str = PROJECT_ROOT) -> str:
        return os.path.join(self._base_dir(project_root), 'prepare_data')

    def results_dir(self, project_root: str = PROJECT_ROOT) -> str:
        return os.path.join(self._base_dir(project_root), 'results')

    def cf_run_dir(self, project_root: str = PROJECT_ROOT) -> str:
        """冻结 CF 模型所在目录：cf_run 是具体 run 就用它，是父目录就按 seed 匹配。"""
        base = self.cf_run if os.path.isabs(self.cf_run) else os.path.join(project_root, self.cf_run)
        if os.path.exists(os.path.join(base, 'config.json')):
            return base

        candidates = []
        for d in sorted(glob.glob(os.path.join(base, '*'))):
            cfg_path = os.path.join(d, 'config.json')
            if not os.path.isdir(d) or not os.path.exists(cfg_path):
                continue
            with open(cfg_path, encoding='utf-8') as f:
                if json.load(f).get('seed') == self.seed:
                    candidates.append(d)

        if not candidates:
            raise FileNotFoundError(f'{base} 下没有 seed={self.seed} 的 CF run')
        if len(candidates) > 1:
            candidates.sort(key=os.path.getmtime)
            print(f'[warn] {base} 下有 {len(candidates)} 个 seed={self.seed} 的 run，'
                  f'用最新的 {os.path.basename(candidates[-1])}')
        return candidates[-1]

    def run_dir(self, project_root: str = PROJECT_ROOT) -> str:
        return os.path.join(self.results_dir(project_root), self.run_id)

    def create_run_dir(self, project_root: str = PROJECT_ROOT) -> str:
        """创建（或复用）本次实验目录，并把 config.json 写进去。"""
        run_dir = self.run_dir(project_root)
        if self.exist and os.path.exists(run_dir):
            shutil.rmtree(run_dir)
            print(f'removed existing run dir: {run_dir}')
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(self.prepare_data_dir(project_root), exist_ok=True)
        with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, default=str)
        return run_dir

    @classmethod
    def from_dict(cls, values) -> 'AlignConfig':
        """从 dict 构造 AlignConfig，忽略未知字段。"""
        values = dict(values)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})

    @classmethod
    def from_json(cls, path: str) -> 'AlignConfig':
        with open(path, encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def load(cls, run: str, project_root: str = PROJECT_ROOT, save_root: str = None) -> 'AlignConfig':
        """按 run 恢复 config：支持 config.json 路径 / run 目录 / run_id（md5）。"""
        path = run
        if not os.path.exists(path):
            base = save_root or cls().save_root
            if not os.path.isabs(base):
                base = os.path.join(project_root, base)
            matches = sorted(glob.glob(os.path.join(base, '*', 'results', run)))
            if not matches:
                matches = [p for p in [os.path.join(base, run)] if os.path.exists(p)]
            if not matches:
                raise FileNotFoundError(f'run not found: {run} under {base}')
            path = matches[0]
        if os.path.isdir(path):
            path = os.path.join(path, 'config.json')
        return cls.from_json(path)


def iter_run_dirs(project_root: str = PROJECT_ROOT, save_root: str = None) -> list:
    """列出 <save_root>/*/results/* 下所有含 config.json 的 run 目录。"""
    base = save_root or AlignConfig().save_root
    if not os.path.isabs(base):
        base = os.path.join(project_root, base)
    pattern = os.path.join(base, '*', 'results', '*')
    return [d for d in sorted(glob.glob(pattern))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, 'config.json'))]
