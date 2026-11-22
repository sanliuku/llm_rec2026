"""LLaRA_old：用 sasrec_old 的 concept 数据 + 论文原版 NLL（旧工程 train_mooc_nll.py 的口径）。

用法：
    python LLaRA_old/train_mooc_llara_old.py train
    python LLaRA_old/train_mooc_llara_old.py run --num_epochs 3 --batch_size 12
    python LLaRA_old/train_mooc_llara_old.py sweep
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

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_LLARA_DIR = os.path.join(os.path.dirname(_CUR_DIR), 'LLaRA')
# sys.path[0] 放 LLaRA_old（config.py 用本目录的），reasoner / prompt 复用 LLaRA 的同名模块
sys.path.insert(0, _LLARA_DIR)
sys.path.insert(0, _CUR_DIR)

from config import Config, PROJECT_ROOT
from prompt import generate_prompt, get_prompt
from reasoner import LlaRA


def _abs(path: str) -> str:
    """相对路径按项目根目录解析。"""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(PROJECT_ROOT, path))


# ---------------------------------------------------------------------------
# 课程调度器（式 9 / 10）：p(τ) = (τ/T)^γ，按「已消费样本数」离散化 τ
# ---------------------------------------------------------------------------
class CurriculumScheduler:
    def __init__(
        self,
        samples_per_epoch: int,
        p0: float = 0.0,
        p1: float = 1.0,
        gamma: float = 1.0,
        warmup_frac: float = 0.0,
        enabled: bool = True,
        fixed_ratio: float = 1.0,
    ):
        self.total = max(1.0, float(samples_per_epoch))
        self.tau = 0.0
        self.p0 = float(p0)
        self.p1 = float(p1)
        self.gamma = float(gamma)
        self.warmup_frac = float(warmup_frac)
        self.enabled = bool(enabled)
        self.fixed_ratio = float(fixed_ratio)

    @property
    def progress(self) -> float:
        return min(1.0, self.tau / self.total)

    def probability(self) -> float:
        if not self.enabled:
            return self.fixed_ratio
        prog = self.progress
        if self.warmup_frac > 0.0:
            if prog < self.warmup_frac:
                return 0.0
            prog = (prog - self.warmup_frac) / (1.0 - self.warmup_frac)
        p = self.p0 + (self.p1 - self.p0) * (prog ** self.gamma)
        return min(1.0, max(0.0, p))

    def sample_is_hard(self) -> bool:
        is_hard = random.random() < self.probability()
        self.tau += 1.0
        if self.tau >= self.total:
            self.tau = 0.0
        return is_hard


# ---------------------------------------------------------------------------
# 输入构建：concept 数据 -> prompt 样本
# ---------------------------------------------------------------------------
def load_concept_rows(
    data_path: str,
    id2title: Dict[int, str],
    num_his: int,
    target_filter: str = 'none',
) -> List[Dict[str, Any]]:

    with open(data_path, 'rb') as f:
        rows = pkl.load(f)

    if target_filter not in ('none', 'window', 'full'):
        raise ValueError(f"target_filter 只能是 'none'/'window'/'full'，收到 {target_filter!r}")

    n_items = len(id2title)
    # 泄漏比例用前 PROBE 行估算：完整历史最长 5800+，对 49 万行全建 set 要多花一两分钟
    probe_n = min(len(rows), 20000)
    probe_win = probe_full = 0

    data, skipped, dropped, bad_his = [], 0, 0, 0
    for i, row in enumerate(rows):
        his = [int(c) for c in row[1] if c]
        # 越界 id 会让 k_embs / id2title 直接越界；用 min/max 快速判定，正常数据不进分支
        if his and (min(his) < 1 or max(his) > n_items):
            his = [c for c in his if 1 <= c <= n_items]
            bad_his += 1
        pid = int(row[2])
        if not his or not (1 <= pid <= n_items):
            skipped += 1
            continue
        window = his[-num_his:]
        if i < probe_n:
            probe_win += pid in window
            probe_full += pid in set(his)
        if target_filter == 'window':
            hit = pid in window
        elif target_filter == 'full':
            hit = pid in set(his)
        else:
            hit = False
        if hit:
            dropped += 1
            continue
        data.append({'uid': int(row[0]), 'his_concepts': window, 'pid': pid})

    if bad_his:
        print(f'[data] {bad_his} rows had out-of-range concept ids (dropped those ids)')
    if skipped:
        print(f'[data] skipped {skipped} rows (empty history / bad target)')
    if dropped:
        print(f'[data] dropped {dropped} rows by target_filter={target_filter!r}')
    if probe_n:
        print(f'[data] target 已在历史中（前 {probe_n} 行估算）: '
              f'可见窗口 {probe_win / probe_n:.2%}  完整历史 {probe_full / probe_n:.2%}  '
              f'(target_filter={target_filter!r})')
    return data


class PromptDataset(Dataset):
    """一个样本一次只产出一条序列：按 p(τ) 决定 easy（text-only）还是 hard（hybrid）。"""

    def __init__(
        self,
        datalist: List[Dict[str, Any]],
        k_embs: np.ndarray,
        id2title: Dict[int, str],
        tokenizer: Any,
        config: Config,
        curriculum: CurriculumScheduler,
    ):
        self.datalist = datalist
        self.k_embs = k_embs
        self.id2title = id2title
        self.tokenizer = tokenizer
        self.config = config
        self.curriculum = curriculum
        self.num_items = len(id2title)
        self.instruction, self.history_prompt = get_prompt()
        # 与 reasoner.LlaRA 里 add_tokens 出来的 id 一致；用于核对截断后占位符数量
        self.his_token_id = tokenizer.convert_tokens_to_ids('<his_des>')
        self.concept_token_id = tokenizer.convert_tokens_to_ids('<concept_des>')
        self.sep_ids = tokenizer.encode(config.assistant_sep, add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.datalist)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.build(self.datalist[idx], self.curriculum.sample_is_hard())

    # ---------------- 候选集 ----------------
    def get_candidates(self, pid: int, his_concepts: List[int]) -> List[int]:
        """1 正 + neg_num 负。负样本用拒绝采样，避免每次对全库求差集。"""
        exclude = set(int(c) for c in his_concepts)
        exclude.add(int(pid))

        num_neg = max(0, self.config.neg_num)
        population = self.num_items
        negs: List[int] = []
        seen = set()
        attempts = 0
        max_attempts = num_neg * 50 + 100
        while len(negs) < num_neg and attempts < max_attempts:
            attempts += 1
            cand = random.randrange(population) + 1  # concept id 1-based
            if cand in exclude or cand in seen:
                continue
            seen.add(cand)
            negs.append(cand)
        if len(negs) < num_neg:
            # 极少数情况下（历史几乎覆盖全库）退化为顺序补齐，保证形状稳定
            for cand in range(1, population + 1):
                if len(negs) >= num_neg:
                    break
                if cand not in exclude and cand not in seen:
                    seen.add(cand)
                    negs.append(cand)

        candidates = negs + [int(pid)]
        random.shuffle(candidates)
        return candidates

    # ---------------- 单条样本 ----------------
    def build(self, dp: Dict[str, Any], is_hard: bool) -> Dict[str, Any]:
        his = dp['his_concepts']
        pid = dp['pid']
        candidates = self.get_candidates(pid, his)

        his_titles = [self.id2title[c - 1] for c in his]
        cand_titles = [self.id2title[c - 1] for c in candidates]

        his_ph = ','.join(f'{i + 1}.{t} <PH>' for i, t in enumerate(his_titles))
        his_des = ','.join(f'{i + 1}.{t} <his_des>' for i, t in enumerate(his_titles))
        cand_ph = ', '.join(f'{t} <PH>' for t in cand_titles)
        cand_des = ', '.join(f'({t} <concept_des>)' for t in cand_titles)

        warm_input = self.history_prompt.format(historical_concepts=his_ph, candidates=cand_ph)
        hard_input = self.history_prompt.format(historical_concepts=his_des, candidates=cand_des)

        if is_hard:
            # hybrid prompt：历史/候选用 <his_des> / <concept_des> 占位，由投影层产生向量
            _, user_prompt = generate_prompt(self.instruction, warm_input, hard_input)
        else:
            # text-only prompt：只给 title + 零向量的 <PH> 占位符
            user_prompt, _ = generate_prompt(self.instruction, warm_input, hard_input)

        full_prompt = user_prompt + self.id2title[pid - 1] + self.config.end_sep
        tokenized = self.tokenizer(
            full_prompt,
            truncation=True,
            max_length=self.config.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        input_ids = tokenized['input_ids']

        # response 起点：直接找 `### Response:\n` 的 token 位置，比单独 tokenize 一遍更准
        resp_start = self._response_start(input_ids, user_prompt)
        tokenized['labels'] = [-100] * resp_start + list(input_ids[resp_start:])

        item: Dict[str, Any] = dict(tokenized)
        item['is_hard'] = int(is_hard)
        if is_hard:
            # 左侧截断会从序列头部切 token：活下来的历史占位符是**最后**若干个，
            # 因此 his_embs 也必须同步取最后若干个，否则 get_embs 里按位置填充会 shape 不匹配。
            his_token_cnt = sum(1 for t in input_ids if t == self.his_token_id)
            cand_token_cnt = sum(1 for t in input_ids if t == self.concept_token_id)
            if cand_token_cnt != len(candidates):
                raise ValueError(
                    f'候选被 cutoff_len={self.config.cutoff_len} 截断 '
                    f'({cand_token_cnt}/{len(candidates)})：正例可能已丢，样本无效；'
                    f'请调大 cutoff_len 或减小 num_his/neg_num')
            if his_token_cnt != len(his):
                his = his[len(his) - his_token_cnt:] if his_token_cnt > 0 else []
            item['his_embs'] = torch.from_numpy(
                np.asarray(self.k_embs[[c - 1 for c in his]], dtype=np.float32).reshape(-1, self.k_embs.shape[1]))
            item['candidate_embs'] = torch.from_numpy(
                np.asarray(self.k_embs[[c - 1 for c in candidates]], dtype=np.float32))
        return item

    def _response_start(self, input_ids: List[int], user_prompt: str) -> int:
        sep = self.sep_ids
        last = max(len(input_ids) - 1, 0)  # 至少留一个 token，避免整条被 -100 屏蔽
        for i in range(len(input_ids) - len(sep), -1, -1):
            if input_ids[i:i + len(sep)] == sep:
                return min(i + len(sep), last)
        # 兜底：理论上走不到，除非 prompt 模板和 assistant_sep 不同步
        fallback = len(self.tokenizer(user_prompt, add_special_tokens=False)['input_ids'])
        return min(fallback, last)


# ---------------------------------------------------------------------------
# Collator：只负责 his_embs / candidate_embs 的补齐
# ---------------------------------------------------------------------------
class LLMCollator(DataCollatorForSeq2Seq):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = [dict(f) for f in features]

        his_values = [f.pop('his_embs') for f in features if 'his_embs' in f]
        cand_values = [f.pop('candidate_embs') for f in features if 'candidate_embs' in f]
        is_hard = torch.tensor([f.pop('is_hard', 0) for f in features], dtype=torch.long)

        batch = super().__call__(features)
        batch['is_hard'] = is_hard

        if his_values:
            max_his_len = max(h.shape[0] for h in his_values)
            padded = []
            for h in his_values:
                if h.shape[0] < max_his_len:
                    # 左侧补零，reasoner.get_embs 会按「整行是否为 0」把这些行过滤掉
                    pad = torch.zeros((max_his_len - h.shape[0], h.shape[1]), dtype=h.dtype)
                    h = torch.cat((pad, h), dim=0)
                padded.append(h)
            batch['his_embs'] = torch.stack(padded, dim=0)
        if cand_values:
            batch['candidate_embs'] = torch.stack(cand_values, dim=0)
        return batch


# ---------------------------------------------------------------------------
# Trainer：标准 NLL
# ---------------------------------------------------------------------------
class NLLTrainer(transformers.Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._meter: Dict[str, List[float]] = {}
        # 实际喂进模型的 hard / 全样本数：用于核对课程调度是否真的在推进
        self.hard_seen = 0
        self.total_seen = 0
        self._labels_checked = False

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir, save_embedding_layers=False)
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs['labels']
        is_hard = inputs.get('is_hard')

        # 有效 label 只落在序列尾部（prompt 段与左侧 padding 都是 -100），
        # 所以只算最后 k 个位置：全序列算 CE 会多出 (B, L, vocab) 的 log_softmax 和一份 reshape 副本
        k = max(int((labels != -100).sum(dim=1).max().item()), 1)
        if not self._labels_checked:
            n_valid = (labels != -100).sum(dim=1)
            if not torch.equal((labels[:, -k:] != -100).sum(dim=1), n_valid):
                raise RuntimeError('labels 不是右对齐的，尾部切片会错位；检查 collator 的 padding_side')
            self._labels_checked = True
        outputs = model(inputs, logits_to_keep=k + 1)  # LlaRA.forward 只吃 input_ids / his_embs / candidate_embs
        logits = outputs.logits

        # 因果 LM 的错位：位置 t 的 logits 预测标签 t+1
        shift_logits = logits[:, -(k + 1):-1, :]
        shift_labels = labels[:, -k:]
        vocab_size = shift_logits.size(-1)

        token_loss = F.cross_entropy(
            shift_logits.reshape(-1, vocab_size),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction='none',
        ).view(shift_labels.shape)

        valid = shift_labels != -100
        n_tokens = valid.sum().float().clamp(min=1.0)
        loss = token_loss[valid].sum() / n_tokens

        with torch.no_grad():
            self._meter.setdefault('ntok', []).append(float(n_tokens))
            if is_hard is not None:
                self.hard_seen += int(is_hard.sum())
                self.total_seen += int(is_hard.numel())
                hard = is_hard.to(valid.device).bool().unsqueeze(1).expand_as(valid)
                for tag, mask in (('easy', valid & ~hard), ('hard', valid & hard)):
                    if mask.any():
                        self._meter.setdefault(f'loss_{tag}', []).append(float(token_loss[mask].mean()))
                self._meter.setdefault('hard_ratio', []).append(float(is_hard.float().mean()))

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
    """单用户（batch=1）端到端推理时延：QMoE 拼 embedding + 自回归生成，返回 ms/user。"""
    device = next(model.parameters()).device
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model.llm.generate(
                inputs_embeds=model.get_embs(batch),
                attention_mask=batch['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model.llm.generate(
                inputs_embeds=model.get_embs(batch),
                attention_mask=batch['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1000


def train(config: Optional[Config] = None):
    config = config or Config()

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    transformers.set_seed(config.seed)

    instruction_prompt, history_prompt = get_prompt()
    print(f'[prompt] instruction: {instruction_prompt}')
    print(f'[prompt] history    : {history_prompt}')
    accelerator = Accelerator()

    # ---------------- 数据 ----------------
    concept_csv = os.path.join(config.need_data_dir(PROJECT_ROOT), 'entities/concepts.csv')
    concept_df = pd.read_csv(concept_csv, header=None, names=['index', 'id'])
    num_items = len(concept_df)
    id2title = {int(k): str(v).replace('K_', '') for k, v in zip(concept_df['index'], concept_df['id'])}
    print(f'[data] concepts={num_items}  first title={id2title.get(0)}')

    emb_path = _abs(config.k_emb_path)
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f'Emb_k 不存在: {emb_path}')
    k_embs = np.load(emb_path).astype(np.float32, copy=False)
    if k_embs.shape[0] != num_items:
        raise ValueError(f'Emb_k 行数 {k_embs.shape[0]} 与概念数 {num_items} 不一致: {emb_path}')
    if k_embs.shape[1] != config.embed_size:
        raise ValueError(f'Emb_k 维度 {k_embs.shape[1]} 与 embed_size={config.embed_size} 不一致')
    print(f'[data] Emb_k={k_embs.shape}  <- {emb_path}')

    train_path = _abs(config.train_data_path)
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f'train 数据不存在: {train_path}；先跑 scripts/build_concept_data.py')
    train_data = load_concept_rows(train_path, id2title, config.num_his, config.target_filter)
    if config.debug_samples:
        train_data = train_data[: config.debug_samples]
    print(f'[data] samples={len(train_data)}  <- {train_path}')

    # ---------------- 输出目录 ----------------
    accelerator.wait_for_everyone()
    output_dir = config.create_run_dir(PROJECT_ROOT)

    # ---------------- 模型 ----------------
    model = LlaRA(model_path=config.base_model, config=config, device=config.device, is_lora=True)
    tokenizer = model.tokenizer

    # ---------------- 数据集 ----------------
    # T = 一个 epoch 的样本数：τ 走满一轮就归零，每个 epoch 重跑一遍 easy → hard
    curriculum = CurriculumScheduler(
        samples_per_epoch=len(train_data),
        p0=config.curriculum_p0,
        p1=config.curriculum_p1,
        gamma=config.curriculum_power,
        warmup_frac=config.curriculum_warmup_frac,
        enabled=config.curriculum,
        fixed_ratio=config.fixed_hard_ratio,
    )
    dataset = PromptDataset(train_data, k_embs, id2title, tokenizer, config, curriculum)
    print(f'[curriculum] T={curriculum.total:.0f} 样本/epoch  p(τ)→{config.curriculum_p1}  '
          f'enable={config.curriculum}  gamma={config.curriculum_power}  '
          f'epochs={config.num_epochs}（每 epoch 归零重走）')

    # 冒烟自检：打印一条 hard 样本的 response 段解码结果，确认标签只落在 title 上
    probe = dataset.build(train_data[0], True)
    resp_ids = [t for t, l in zip(probe['input_ids'], probe['labels']) if l != -100]
    print(f"[probe] seq_len={len(probe['input_ids'])}  resp_tokens={len(resp_ids)}  "
          f'resp={tokenizer.decode(resp_ids)!r}')
    assert len(resp_ids) > 0, 'response 段被全部屏蔽，检查 assistant_sep 与 prompt 模板是否一致'

    # ---------------- Trainer ----------------
    collator = LLMCollator(tokenizer, pad_to_multiple_of=config.pad_to_multiple_of)
    # 课程 τ 依赖单进程顺序调用 __getitem__，config 里 dataloader_num_workers=0
    train_args_kwargs: Dict[str, Any] = dict(
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
    )
    # 课程 τ 在本进程里按 epoch 内进度累加并归零，num_workers>0 会被 worker 复制打散
    if config.dataloader_num_workers > 0 and config.curriculum:
        print(f'[warn] dataloader_num_workers={config.dataloader_num_workers} 且启用了课程：'
              f'每个 worker 只经手 1/{config.dataloader_num_workers} 的样本，'
              f'p(τ) 上界降为 {1 / config.dataloader_num_workers:.2f}，课程会显著减弱；'
              f'建议设 dataloader_num_workers=0')
    trainer = NLLTrainer(
        model=model,
        train_dataset=dataset,
        data_collator=collator,
        args=transformers.TrainingArguments(**train_args_kwargs),
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
    hard_seen, total_seen = trainer.hard_seen, trainer.total_seen
    hard_ratio_observed = hard_seen / total_seen if total_seen else 0.0
    print(f'[curriculum] 实际 hard 占比={hard_ratio_observed:.3f} '
          f'({hard_seen}/{total_seen})，每 epoch 归零的线性调度理论值≈0.5')

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
        'curriculum_enabled': bool(config.curriculum),
        'hard_ratio_observed': hard_ratio_observed,
        'target_filter': config.target_filter,
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print('metrics:', json.dumps(metrics, ensure_ascii=False, indent=2))

    model.save_pretrained(output_dir, False)
    print(f'[done] 权重已保存到 {output_dir}')


# 多组实验：每项是 Config 的覆盖字段，按顺序训练；每个 run 按 config md5 落在独立目录。
# 覆盖 sasrec_old 在 mid_data 上的 5 个 seed（Emb_k 与 sasrec_old 的训练 seed 一一对应）。
SWEEP = [
    # {'seed': 12, 'k_emb_path': 'sasrec_old/runs/mid_data/prepare_data/1b179df098b8ad3801d17f65cc105291/Emb_k.npy'},
    {'seed': 22, 'k_emb_path': 'sasrec_old/runs/mid_data/prepare_data/742474b1b5911a90cf0b31d303c78db6/Emb_k.npy'},
    {'seed': 32, 'k_emb_path': 'sasrec_old/runs/mid_data/prepare_data/05b5d096e123b7b89cf4ccfaed200fad/Emb_k.npy'},
    # {'seed': 42, 'k_emb_path': 'sasrec_old/runs/mid_data/prepare_data/116719ce0d1bed7972d4defb6130686c/Emb_k.npy'},
    {'seed': 52, 'k_emb_path': 'sasrec_old/runs/mid_data/prepare_data/d807deb4019e7acafd77f5c197cc2d2f/Emb_k.npy'},
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
    """单次运行：python train_mooc_llara_old.py run --num_epochs 3 --batch_size 12"""
    train(Config(**_cast(overrides)))


def sweep():
    """跑 SWEEP 里的全部配置：python train_mooc_llara_old.py sweep"""
    for i, overrides in enumerate(SWEEP):
        print(f'===== [{i + 1}/{len(SWEEP)}] {overrides or "default"} =====')
        train(Config(**overrides))
        torch.cuda.empty_cache()


if __name__ == '__main__':
    fire.Fire({'train': train, 'run': run, 'sweep': sweep})
