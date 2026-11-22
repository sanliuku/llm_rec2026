"""A-LLMRec 第二阶段（LLM 侧）的实验配置：run 目录由 config 的 md5 命名。

数据来自第一阶段（`ALLMRec_old/prepare_align.py` 导出）::

    ALLMRec_old/runs/<dataset>/prepare_data/<run_id>/Emb_k.npy               # e_i，128 维
    ALLMRec_old/runs/<dataset>/prepare_data/<run_id>/train_user_features.npy # x^u，256 维
    prepare_data/mooc_sasrec_augllm2/train_concept_data.p                    # 行 = [uid, his, target]

模型用本目录的 `pt_reasoner.py::PTReasoner`（与 `ALLMRec/` 那套无关），
与官方 `pre_train_phase2` 一致：LLM 冻结，只训 F_U、F_I 两个投影层，损失是回答段的标准 NLL。
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
    # dataset_dir: str = 'data/MOOCCube/mid_data'
    dataset_dir: str = 'data/MOOCCube/small_data'
    train_data_path: str = 'prepare_data/mooc_sasrec_augllm/train_concept_data.p'
    k_emb_path: str = ('ALLMRec_old/runs/small_data/prepare_data/'
                       '3cd4865a871d47fa9f4c521b1d4f9443/Emb_k.npy')  # stage-1 seed=42
    u_emb_path: str = ''
    num_his: int = 20  
    debug_samples: int = 0  

    # ---- 模型 ----
    base_model: str = '../../local_llm/Qwen2.5-1.5B'
    user_embed_size: int = 256
    item_embed_size: int = 128
    token_size: int = 1536  
    user_act: str = 'leakyrelu'  # 官方 log_emb_proj 的激活
    item_act: str = 'gelu'       # 官方 item_emb_proj 的激活

    # ---- lora ----
    use_lora: bool = False  # 官方为 False：LLM 全冻结
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        'q_proj', 'v_proj'])

    # ---- 训练 ----
    seed: int = 42
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 20
    lr_scheduler_type: str = 'linear'
    bf16: bool = True
    tf32: bool = True
    optim: str = 'adamw_torch'
    max_steps: int = -1  
    logging_steps: int = 10
    save_steps: int = 20000
    save_total_limit: int = 2
    dataloader_num_workers: int = 0
    pad_to_multiple_of: int = 8
    device: str = 'cuda:0'

    # ---- 采样 ----
    neg_num: int = 19  # 每个正例配的负例数（官方 candidate_num=20，即 1 正 + 19 负）

    # ---- 评测 ----
    latency_repeats: int = 20  # 推理时延测量的重复次数（batch=1）
    latency_max_new_tokens: int = 16  # 时延测量时生成的 token 上限

    # ---- 提示 ----
    assistant_sep: str = '### Response:\n'
    end_sep: str = '<|endoftext|>'

    # ---- 推理（model.load_model 用）----
    lora_weights_path: str = '' 

    # ---- 输出 ----
    save_root: str = 'ALLMRec_llm/runs'
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

    def need_data_dir(self, project_root: str = PROJECT_ROOT) -> str:
        """原始实体/关系表目录（entities、relations）。"""
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

    def resolve_stage1_emb(self, project_root: str = PROJECT_ROOT) -> tuple:
        """返回 (Emb_k, train_user_features) 的绝对路径。

        k_emb_path 必填（各 seed 的产物在同一个 prepare_data 下的不同 run_id 目录里，
        无法自动判断要哪个 seed）；u_emb_path 留空时取同目录的 train_user_features.npy，
        保证 x^u 与 e_i 来自同一次 stage-1 训练。
        """
        def _abs(p):
            return p if os.path.isabs(p) else os.path.normpath(os.path.join(project_root, p))

        if not self.k_emb_path:
            pattern = os.path.join(project_root, 'ALLMRec_old', 'runs', '*',
                                   'prepare_data', '*', 'Emb_k.npy')
            hits = sorted(glob.glob(pattern))
            raise FileNotFoundError(
                f'未填 k_emb_path；现有 stage-1 产物 {len(hits)} 份（按 seed 选择）：\n  ' +
                '\n  '.join(hits or ['(空，先跑 ALLMRec_old/prepare_align.py)']))

        k_path = _abs(self.k_emb_path)
        u_path = _abs(self.u_emb_path) if self.u_emb_path else \
            os.path.join(os.path.dirname(k_path), 'train_user_features.npy')
        for path in (k_path, u_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f'stage-1 产物不存在: {path}')
        return k_path, u_path

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
