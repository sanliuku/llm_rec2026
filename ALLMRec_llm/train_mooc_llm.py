"""第二阶段的训练：冻结 LLM，只训 F_U / F_I 两个投影层，损失用回答段的标准 NLL。
用法：
    python ALLMRec_llm/train_mooc_llm.py run --num_epochs 1 --batch_size 4
    python ALLMRec_llm/train_mooc_llm.py sweep
"""

import gc
import json
import os
import pickle as pkl
import random
import sys
import time
from typing import Any, Dict, List, Optional

import fire
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, PROJECT_ROOT
from prompt import build_input, generate_prompt, get_prompt
from pt_reasoner import PTReasoner


def _abs(path: str) -> str:
    """相对路径按项目根目录解析。"""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(PROJECT_ROOT, path))


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_concept_rows(rows: List[Any], num_items: int) -> List[Dict[str, Any]]:
    """`.p` 的每行转成样本，并记住原始行号（x^u 是按行对齐的，不能丢）。

    concept id 统一 1-based：his_concepts 去掉 padding 后按时间顺序，pid 是目标 concept。
    越界 id 直接剔除；历史为空或 target 越界的行跳过（行号保留，不影响 user_emb 取值）。
    """
    data, skipped, bad_his = [], 0, 0
    for i, row in enumerate(rows):
        his = [int(c) for c in row[1] if c]
        if his and (min(his) < 1 or max(his) > num_items):
            his = [c for c in his if 1 <= c <= num_items]
            bad_his += 1
        pid = int(row[2])
        if not his or not (1 <= pid <= num_items):
            skipped += 1
            continue
        data.append({'row': i, 'uid': int(row[0]), 'his_concepts': his, 'pid': pid})

    if bad_his:
        print(f'[data] {bad_his} 行含越界 concept id（已剔除这些 id）')
    if skipped:
        print(f'[data] 跳过 {skipped} 行（历史为空 / target 越界）')
    return data


class PromptDataset(Dataset):
    """一条样本 = prompt + 候选 + 三个占位符对应的向量。"""

    def __init__(self, datalist, id2title, tokenizer, config, k_embs, u_embs):
        self.datalist = datalist
        self.id2title = id2title
        self.tokenizer = tokenizer
        self.config = config
        self.k_embs = k_embs
        self.u_embs = u_embs
        self.num_items = len(id2title)
        self.instruction, self.history_prompt = get_prompt()
        self.his_token_id = tokenizer.convert_tokens_to_ids('<his_des>')
        self.sep_ids = tokenizer.encode(config.assistant_sep, add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.datalist)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.build(self.datalist[idx])

    # ---------------- 候选集 ----------------
    def get_candidates(self, pid: int, his_concepts: List[int]) -> List[int]:
        """1 正 + neg_num 负（官方 candidate_num=20）。负例拒绝采样，避免对全库求差集。"""
        exclude = set(int(c) for c in his_concepts)
        exclude.add(int(pid))
        negs: List[int] = []
        seen = set()
        attempts = 0
        num_neg = max(0, self.config.neg_num)
        max_attempts = num_neg * 50 + 100
        while len(negs) < num_neg and attempts < max_attempts:
            attempts += 1
            cand = random.randrange(self.num_items) + 1  # concept id 1-based
            if cand in exclude or cand in seen:
                continue
            seen.add(cand)
            negs.append(cand)
        if len(negs) < num_neg:  # 历史几乎覆盖全库时顺序补齐，保证候选数稳定
            for cand in range(1, self.num_items + 1):
                if len(negs) >= num_neg:
                    break
                if cand not in exclude and cand not in seen:
                    seen.add(cand)
                    negs.append(cand)
        candidates = negs + [int(pid)]
        random.shuffle(candidates)
        return candidates

    # ---------------- 单条样本 ----------------
    def build(self, dp: Dict[str, Any]) -> Dict[str, Any]:
        his = dp['his_concepts'][-self.config.num_his:]
        pid = dp['pid']
        candidates = self.get_candidates(pid, his)

        input_text = build_input(his, candidates, self.id2title, self.history_prompt)
        user_prompt = generate_prompt(self.instruction, input_text)
        full_prompt = user_prompt + self.id2title[pid - 1] + self.config.end_sep

        # 官方 truncation=False，这里同样不截断
        tokenized = self.tokenizer(full_prompt, padding=False, return_tensors=None)
        input_ids = tokenized['input_ids']
        resp_start = self._response_start(input_ids, user_prompt)
        tokenized['labels'] = [-100] * resp_start + list(input_ids[resp_start:])
        assert input_ids.count(self.his_token_id) == len(his), '历史占位符数与向量行数不一致'

        item: Dict[str, Any] = dict(tokenized)
        item['user_emb'] = torch.from_numpy(self.u_embs[dp['row']].astype(np.float32))
        item['his_embs'] = torch.from_numpy(
            np.asarray(self.k_embs[[c - 1 for c in his]], dtype=np.float32)
            .reshape(-1, self.k_embs.shape[1]))
        item['candidate_embs'] = torch.from_numpy(
            np.asarray(self.k_embs[[c - 1 for c in candidates]], dtype=np.float32))
        return item

    def _response_start(self, input_ids: List[int], user_prompt: str) -> int:
        """response 起点 = `### Response:\\n` 之后第一个 token。"""
        sep = self.sep_ids
        last = max(len(input_ids) - 1, 0)  # 至少留一个 token，避免整条被 -100 屏蔽
        for i in range(len(input_ids) - len(sep), -1, -1):
            if input_ids[i:i + len(sep)] == sep:
                return min(i + len(sep), last)
        fallback = len(self.tokenizer(user_prompt, add_special_tokens=False)['input_ids'])
        return min(fallback, last)


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------
class CandidateCollator(DataCollatorForSeq2Seq):
    """补 his_embs：候选数固定、用户每条一个，只有历史长度需要左侧补零。"""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = [dict(f) for f in features]
        user_values = [f.pop('user_emb') for f in features]
        his_values = [f.pop('his_embs') for f in features]
        cand_values = [f.pop('candidate_embs') for f in features]

        batch = super().__call__(features)

        max_his_len = max(h.shape[0] for h in his_values)
        padded = []
        for h in his_values:
            if h.shape[0] < max_his_len:
                # 左侧补零：get_embs 按「整行是否为 0」过滤，顺序不会乱
                pad = torch.zeros((max_his_len - h.shape[0], h.shape[1]), dtype=h.dtype)
                h = torch.cat((pad, h), dim=0)
            padded.append(h)
        batch['user_emb'] = torch.stack(user_values, dim=0)
        batch['his_embs'] = torch.stack(padded, dim=0)
        batch['candidate_embs'] = torch.stack(cand_values, dim=0)
        return batch


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class NLLTrainer(transformers.Trainer):
    """answer 段全词表交叉熵，prompt 段 -100 屏蔽，与官方 `outputs.loss` 等价。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._meter: Dict[str, List[float]] = {}
        self._labels_checked = False

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir, save_embedding_layers=False)
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs['labels']
        # 有效 label 只落在序列尾部（prompt 段与左侧 padding 都是 -100），
        # 所以只算最后 k 个位置：全序列算 CE 会多出 (B, L, vocab) 的 log_softmax 和一份 reshape 副本
        k = max(int((labels != -100).sum(dim=1).max().item()), 1)
        if not self._labels_checked:
            n_valid = (labels != -100).sum(dim=1)
            if not torch.equal((labels[:, -k:] != -100).sum(dim=1), n_valid):
                raise RuntimeError('labels 不是右对齐的，尾部切片会错位；检查 collator 的 padding_side')
            self._labels_checked = True
        outputs = model(inputs, logits_to_keep=k + 1)
        logits = outputs.logits

        shift_logits = logits[:, -(k + 1):-1, :]
        shift_labels = labels[:, -k:]
        vocab_size = shift_logits.size(-1)

        token_loss = F.cross_entropy(
            shift_logits.reshape(-1, vocab_size),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction='none',
        )

        valid = shift_labels.reshape(-1) != -100
        n_tokens = valid.sum().float().clamp(min=1.0)
        loss = token_loss[valid].sum() / n_tokens

        with torch.no_grad():
            self._meter.setdefault('ntok', []).append(float(n_tokens))
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs):
        if self._meter:
            for key, values in self._meter.items():
                logs[key] = float(np.mean(values))
            self._meter = {}
        return super().log(logs, *args, **kwargs)


def _gpu_peak_gb():
    return torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0


def _measure_infer_latency(model, batch, repeats=20, max_new_tokens=16, warmup=3):
    """单用户（batch=1）端到端推理时延：拼 embedding + 自回归生成，返回 ms/user。"""
    device = next(model.parameters()).device
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    model.eval()

    def _run_once():
        return model.llm.generate(
            inputs_embeds=model.get_embs(batch),
            attention_mask=batch['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    with torch.no_grad():
        for _ in range(warmup):
            _run_once()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _run_once()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1000


def train(config: Optional[Config] = None):
    config = config or Config()

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    transformers.set_seed(config.seed)

    accelerator = Accelerator()

    # ---------------- 数据 ----------------
    concept_csv = os.path.join(config.need_data_dir(PROJECT_ROOT), 'entities/concepts.csv')
    concept_df = pd.read_csv(concept_csv, header=None, names=['index', 'id'])
    num_items = len(concept_df)
    id2title = {int(k): str(v).replace('K_', '') for k, v in zip(concept_df['index'], concept_df['id'])}
    print(f'[data] concepts={num_items}  first title={id2title.get(0)}')

    k_path, u_path = config.resolve_stage1_emb(PROJECT_ROOT)
    k_embs = np.load(k_path).astype(np.float32, copy=False)
    u_embs = np.load(u_path).astype(np.float32, copy=False)
    if k_embs.shape[0] != num_items:
        raise ValueError(f'Emb_k 行数 {k_embs.shape[0]} 与概念数 {num_items} 不一致: {k_path}')
    if k_embs.shape[1] != config.item_embed_size:
        raise ValueError(f'Emb_k 维度 {k_embs.shape[1]} 与 item_embed_size={config.item_embed_size} 不一致')
    if u_embs.shape[1] != config.user_embed_size:
        raise ValueError(f'user_features 维度 {u_embs.shape[1]} 与 user_embed_size={config.user_embed_size} 不一致')
    print(f'[data] e_i={k_embs.shape}  x^u={u_embs.shape}')
    print(f'[data]   e_i <- {k_path}\n[data]   x^u <- {u_path}')

    train_path = _abs(config.train_data_path)
    if not os.path.exists(train_path):
        raise FileNotFoundError(f'train 数据不存在: {train_path}')
    with open(train_path, 'rb') as f:
        rows = pkl.load(f)
    if u_embs.shape[0] != len(rows):
        raise ValueError(
            f'user_features 行数 {u_embs.shape[0]} 与 .p 行数 {len(rows)} 不一致：'
            f'x^u 必须按 .p 的行一一对齐，别混用不同 run 的产物')
    train_data = load_concept_rows(rows, num_items)
    del rows
    if config.debug_samples:
        train_data = train_data[: config.debug_samples]
    print(f'[data] samples={len(train_data)}  <- {train_path}')

    # ---------------- 输出目录 ----------------
    accelerator.wait_for_everyone()
    output_dir = config.create_run_dir(PROJECT_ROOT)

    # ---------------- 模型 ----------------
    model = PTReasoner(model_path=config.base_model, config=config,
                       device=config.device, is_lora=config.use_lora)
    tokenizer = model.tokenizer

    dataset = PromptDataset(train_data, id2title, tokenizer, config, k_embs, u_embs)

    # 冒烟自检：标签只应落在 title + end_sep 上
    probe = dataset.build(train_data[0])
    resp_ids = [t for t, l in zip(probe['input_ids'], probe['labels']) if l != -100]
    print(f"[probe] seq_len={len(probe['input_ids'])}  resp_tokens={len(resp_ids)}  "
          f'resp={tokenizer.decode(resp_ids)!r}')
    assert len(resp_ids) > 0, 'response 段被全部屏蔽，检查 assistant_sep 与 prompt 模板是否一致'

    collator = CandidateCollator(tokenizer, pad_to_multiple_of=config.pad_to_multiple_of)
    trainer = NLLTrainer(
        model=model,
        train_dataset=dataset,
        data_collator=collator,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            num_train_epochs=config.num_epochs,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            bf16=config.bf16,
            tf32=config.tf32,
            optim=config.optim,
            logging_strategy='steps',
            logging_steps=config.logging_steps,
            save_strategy='steps',
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            save_on_each_node=False,
            log_on_each_node=False,
            dataloader_num_workers=config.dataloader_num_workers,
            ddp_find_unused_parameters=None,
            local_rank=int(os.environ.get('LOCAL_RANK', -1)),
            seed=config.seed,
            data_seed=config.seed,
            remove_unused_columns=False,
            report_to='none',
        ),
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'params: total={total_params:,} trainable={trainable_params:,} '
          f'({100 * trainable_params / total_params:.2f}%)')

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_start = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - train_start
    train_peak = _gpu_peak_gb()
    train_steps = int(trainer.state.global_step)

    # 释放 optimizer 等训练态后再测推理峰值，避免把训练残留算进去
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    infer_latency = _measure_infer_latency(
        model, collator([probe]),
        repeats=config.latency_repeats, max_new_tokens=config.latency_max_new_tokens)
    infer_peak = _gpu_peak_gb()

    metrics = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'trainable_ratio': trainable_params / total_params,
        'train_steps': train_steps,
        'train_time_sec': train_time,
        'train_time_hours': train_time / 3600,
        'train_peak_mem_gb': train_peak,
        'infer_latency_ms_per_user': infer_latency,
        'infer_peak_mem_gb': infer_peak,
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print('metrics:', json.dumps(metrics, ensure_ascii=False, indent=2))

    model.save_pretrained(output_dir, False)
    print(f'[done] 权重已保存到 {output_dir}')


# 多组实验：每项是 Config 的覆盖字段，按顺序训练，每个 run 按 config md5 落在独立目录。
# 覆盖第一阶段在 mid_data 上的 5 个 seed，k_emb_path 与训练 seed 一一对应。
SWEEP = [
    # {'seed': 12, 'k_emb_path': 'ALLMRec_old/runs/mid_data/prepare_data/fe6f92622c461b2216f8bf1a215da13f/Emb_k.npy'},
    # {'seed': 22, 'k_emb_path': 'ALLMRec_old/runs/mid_data/prepare_data/6fa7f48b48ac46366199e773be47fca7/Emb_k.npy'},
    # {'seed': 32, 'k_emb_path': 'ALLMRec_old/runs/mid_data/prepare_data/e55a6c142a6a3366f7a169d0bdacaf7c/Emb_k.npy'},
    # {'seed': 42, 'k_emb_path': 'ALLMRec_old/runs/mid_data/prepare_data/877d74f25f2e7384ab16648ec9b3c997/Emb_k.npy'},
    # {'seed': 52, 'k_emb_path': 'ALLMRec_old/runs/mid_data/prepare_data/a2d4f1011d91615d7ddc8f7478551f53/Emb_k.npy'},
    {'seed': 12, 'k_emb_path': 'ALLMRec_old/runs/small_data/prepare_data/14fa85141c1899b3888451faf6dc146d/Emb_k.npy'},
    {'seed': 22, 'k_emb_path': 'ALLMRec_old/runs/small_data/prepare_data/9a4320be14bd686015d193ca4fe14b69/Emb_k.npy'},
    {'seed': 32, 'k_emb_path': 'ALLMRec_old/runs/small_data/prepare_data/5bcdcced83d6d067651637c29056f320/Emb_k.npy'},
    {'seed': 42, 'k_emb_path': 'ALLMRec_old/runs/small_data/prepare_data/3cd4865a871d47fa9f4c521b1d4f9443/Emb_k.npy'},
    {'seed': 52, 'k_emb_path': 'ALLMRec_old/runs/small_data/prepare_data/08dae6f88e5328559dcf58046a4bf768/Emb_k.npy'},
]


def _cast(overrides):
    """把 fire 传进来的字符串按 Config 字段默认值的类型还原。"""
    default = Config()
    values = {}
    for key, value in overrides.items():
        if not hasattr(default, key):
            raise KeyError(f'unknown config field: {key}')
        cur = getattr(default, key)
        if isinstance(cur, bool):
            values[key] = str(value).lower() in ('1', 'true', 'yes', 'y')
        elif isinstance(cur, int):
            values[key] = int(value)
        elif isinstance(cur, float):
            values[key] = float(value)
        else:
            values[key] = value
    return values


def run(**overrides):
    """单次运行：python ALLMRec_llm/train_mooc_llm.py run --num_epochs 1 --batch_size 4"""
    train(Config(**_cast(overrides)))


def sweep():
    """跑 SWEEP 里的全部配置：python ALLMRec_llm/train_mooc_llm.py sweep"""
    for i, overrides in enumerate(SWEEP):
        print(f'===== [{i + 1}/{len(SWEEP)}] {overrides or "default"} =====')
        train(Config(**overrides))
        torch.cuda.empty_cache()


if __name__ == '__main__':
    fire.Fire({'run': run, 'sweep': sweep})
