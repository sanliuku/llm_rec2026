"""SAKCRec_v3 实验配置：所有可调参数集中在这里，run 目录由 config 的 md5 命名。

目录结构::

    <save_root>/<dataset>/prepare_data/          # 预处理产物（暂时空着）
    <save_root>/<dataset>/results/<md5>/         # 一次实验的全部产物

用法（train 中）::

    config = Config(num_epochs=1, batch_size=4)
    train(config)

同一个 config 会得到同一个 run 目录，里面的 config.json 记录本次实验的全部参数，
checkpoint / 日志 / 指标都写在该目录下。
"""

import glob
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field, fields
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    # ---- 数据 ----
    dataset_dir: str = 'data/MOOCCube/small_data'
    train_data_path: str = 'prepare_data/train_data_new.p'
    u_emb_path: str = 'prepare_data/mooc_sasrec_augllm/train_user_features_new.npy'
    v_emb_path: str = 'prepare_data/mooc_sasrec_augllm/Emb_v.npy'
    k_emb_path: str = 'prepare_data/mooc_sasrec_augllm/Emb_k.npy'
    num_his: int = 10  # 拼进 prompt 的历史视频条数（取最近 num_his 个）
    sample: int = 400000

    # ---- 模型 ----
    base_model: str = '/home/wbbmin/local_llm/Qwen2.5-1.5B'
    embed_size: int = 256
    token_size: int = 1536  # 896 1536 2048 3584
    hidden_size: int = 1536
    dropout_rate: float = 0.1

    # ---- lora ----
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        'q_proj', 'v_proj'])

    # ---- graph ----
    g_topk: int = 3

    # ---- 训练 ----
    seed: int = 42
    batch_size: int = 12
    num_epochs: int = 3
    learning_rate: float = 1e-4
    cutoff_len: int = 1024
    warmup_steps: int = 20
    save_steps: int = 20000
    logging_steps: float = 0.001
    bf16: bool = True
    tf32: bool = True
    optim: str = 'adamw_torch'
    pad_to_multiple_of: int = 8
    device: str = 'cuda:0'

    # ---- 采样 ----
    llm_negative_sample_size: int = 19  # 每个正例配的负例数（候选共 neg_num + 1 个）
    warm_p: float = 0.3  # 用纯文本历史（warm-up）的样本比例

    # ---- loss ----
    tau: float = 1.5
    eta: float = 0.25
    alpha: float = 0.05
    warm_up: int = -1
    tau_min: float = 1.5  # 自适应 tau 的下界
    tau_max: float = 5.0  # 自适应 tau 的上界
    resource_loss_weight: float = 0.3  # 图（video）损失在总 loss 中的权重

    # ---- 评测 ----
    latency_repeats: int = 20  # 推理时延测量的重复次数（batch=1）
    latency_max_new_tokens: int = 16  # 时延测量时生成的 token 上限

    # ---- 提示 ----
    assistant_sep: str = '### Response:\n'
    end_sep: str = '<|endoftext|>'

    # ---- 输出 ----
    save_root: str = 'SAKCRec_v3/runs'
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
        """数据集名，取自 dataset_dir 的最后一级目录名（如 small_data）。"""
        return os.path.basename(os.path.normpath(self.dataset_dir))

    def _base_dir(self, project_root: str = PROJECT_ROOT) -> str:
        base = self.save_root if os.path.isabs(self.save_root) else os.path.join(project_root, self.save_root)
        return os.path.join(base, self.dataset_name)

    def need_data_dir(self, project_root: str = PROJECT_ROOT) -> str:
        """原始实体/关系表目录（entities、relations、matrix）。"""
        return os.path.join(project_root, self.dataset_dir, 'need_data')

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
        """从 dict 构造 Config，忽略未知字段。"""
        values = dict(values)
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
    base = save_root or Config().save_root
    if not os.path.isabs(base):
        base = os.path.join(project_root, base)
    pattern = os.path.join(base, '*', 'results', '*')
    return [d for d in sorted(glob.glob(pattern))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, 'config.json'))]
