"""实验配置：所有可调参数集中在这里，run 目录由 config 的 md5 命名。

目录结构::

    <save_root>/<dataset>/prepare_data/          # 预处理产物（暂时空着）
    <save_root>/<dataset>/results/<md5>/         # 一次实验的全部产物

用法（main 中）::

    config = Config(emb_dim=256, max_len=50, epochs=200)
    main(config)

同一个 config 会得到同一个 run 目录，里面的 config.json 记录本次实验的全部参数，
checkpoint / 日志 / 指标都写在该目录下。
"""

import glob
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field, fields
from typing import Tuple

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    # ---- 数据 ----
    dataset_dir: str = 'data/MOOCCube/mid_data'
    train_date_time: str = '2018-01-01 00:00:00'
    test_date_start: str = '2018-07-01 00:00:00'
    test_date_end: str = '2018-07-15 00:00:00'

    # ---- 模型 ----
    bert_dim: int = 1024
    emb_dim: int = 256
    max_len: int = 50
    dropout_rate: float = 0.1
    num_heads: int = 1
    num_blocks: int = 2

    # ---- 训练 ----
    seed: int = 42
    epochs: int = 200
    early_stop: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-6
    alphas: Tuple[float, ...] = (0.2,)  # alpha 给 video loss，(1-alpha) 给 concept BPR
    shuffle_seq: bool = False
    eval_interval: int = 10
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # ---- 输出 ----
    save_root: str = 'sasrec_new/runs'
    # 目录已存在时是否清空重建（True=替换，False=复用追加）；不参与 md5，不写进 config.json
    exist: bool = field(default=False, metadata={'no_hash': True})

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
        """数据集名，取自 dataset_dir 的最后一级目录名（如 mid_data）。"""
        return os.path.basename(os.path.normpath(self.dataset_dir))

    def _base_dir(self, project_root: str = PROJECT_ROOT) -> str:
        base = self.save_root if os.path.isabs(self.save_root) else os.path.join(project_root, self.save_root)
        return os.path.join(base, self.dataset_name)

    def prepare_data_dir(self, project_root: str = PROJECT_ROOT) -> str:
        return os.path.join(self._base_dir(project_root), 'prepare_data')

    def results_dir(self, project_root: str = PROJECT_ROOT) -> str:
        return os.path.join(self._base_dir(project_root), 'results')

    def run_dir(self, project_root: str = PROJECT_ROOT) -> str:
        """本次实验的目录路径（不创建目录）。"""
        return os.path.join(self.results_dir(project_root), self.run_id)

    def create_run_dir(self, project_root: str = PROJECT_ROOT) -> str:
        """创建（或复用）本次实验目录，并把 config.json 写进去。

        exist=True 时若目录已存在则整个清空重建；exist=False 时复用原目录（日志/指标追加）。
        """
        run_dir = self.run_dir(project_root)
        if self.exist and os.path.exists(run_dir):
            shutil.rmtree(run_dir)
            print(f'removed existing run dir: {run_dir}')
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(self.prepare_data_dir(project_root), exist_ok=True)
        with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, default=str)
        return run_dir

    # ------------------------------------------------------------------ 加载
    @classmethod
    def from_dict(cls, values) -> 'Config':
        """从 dict 构造 Config，忽略未知字段，并把 list 还原成 tuple。"""
        values = dict(values)
        if isinstance(values.get('alphas'), list):
            values['alphas'] = tuple(values['alphas'])
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})

    @classmethod
    def from_json(cls, path: str) -> 'Config':
        with open(path, encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def load(cls, run: str, project_root: str = PROJECT_ROOT, save_root: str = None) -> 'Config':
        """按 run 恢复 config。

        run 可以是：
        - config.json 的完整路径
        - run 目录的完整路径（目录内含 config.json）
        - run_id（md5），此时在 <save_root>/*/results/<run_id>/config.json 中查找
        """
        path = run
        if not os.path.exists(path):
            base = save_root or cls().save_root
            if not os.path.isabs(base):
                base = os.path.join(project_root, base)
            matches = sorted(glob.glob(os.path.join(base, '*', 'results', run)))
            if not matches:  # 兼容旧的 <save_root>/<run_id> 布局
                matches = [p for p in [os.path.join(base, run)] if os.path.exists(p)]
            if not matches:
                raise FileNotFoundError(f'run not found: {run} under {base}')
            path = matches[0]
        if os.path.isdir(path):
            path = os.path.join(path, 'config.json')
        return cls.from_json(path)


def iter_run_dirs(project_root: str = PROJECT_ROOT, save_root: str = None) -> list:
    """列出 <save_root>/*/results/* 下所有含 config.json 的 run 目录。"""
    base = save_root or Config().save_root
    if not os.path.isabs(base):
        base = os.path.join(project_root, base)
    pattern = os.path.join(base, '*', 'results', '*')
    return [d for d in sorted(glob.glob(pattern))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, 'config.json'))]
