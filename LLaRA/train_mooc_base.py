import copy
import gc
import json
import os
import pickle as pkl
import random
import sys
import time
from typing import Optional
import fire
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import pandas as pd
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
import math

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_CUR_DIR))
sys.path.insert(0, _CUR_DIR)

from config import Config, PROJECT_ROOT
from prompt import get_prompt, generate_list_from_pkl, generate_prompt
from reasoner import LlaRA
from Trie import Trie


class CustomTrainer(transformers.Trainer):
    def __init__(
        self,
        *args,
        tau,
        eta,
        warm_up,
        first_valid_token_num,
        alpha,
        tau_min,
        tau_max,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eta = eta
        self.tau = tau
        self.ada_tau = tau
        self.warm_up = warm_up
        self.first_valid_token_num = first_valid_token_num
        self.alpha = alpha
        self.tau_min = tau_min
        self.tau_max = tau_max

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(output_dir, save_embedding_layers=False)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def gather_valid_data(self, local_tensor):
        local_length = torch.tensor([len(local_tensor)], device=local_tensor.device)
        all_lengths = self.accelerator.gather(local_length)
        max_length = all_lengths.max().item()

        padded_local_tensor = torch.zeros(max_length, device=local_tensor.device)
        padded_local_tensor[: len(local_tensor)] = local_tensor
        all_padded_tensor = self.accelerator.gather(padded_local_tensor)

        num_devices = all_lengths.size(0)
        all_padded_tensor = all_padded_tensor.view(num_devices, max_length)

        mask = torch.arange(max_length, device=local_tensor.device).unsqueeze(0) < all_lengths.unsqueeze(1)
        valid_data = all_padded_tensor[mask]

        return valid_data

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        current_step = self.state.global_step
        constrain_mask = inputs.pop("constrain_mask")

        labels = inputs["labels"]
        outputs = model(inputs)
        logits = outputs.logits

        ologits = logits[:, -constrain_mask.size(1) - 1 :, :]
        olabels = labels[:, -constrain_mask.size(1) - 1 :]

        loss = 0
        # Shift so that tokens < n predict n
        shift_logits = ologits[..., :-1, :].contiguous()
        shift_constrain_mask = constrain_mask.contiguous()
        shift_labels = olabels[..., 1:].contiguous()

        shift_logits_cp = shift_logits.detach().clone()
        shift_logits[shift_constrain_mask == 0] = float("-inf")

        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction="none")
        shift_logits = shift_logits.view(-1, self.model.llm.config.vocab_size)
        shift_logits_cp = shift_logits_cp.view(-1, self.model.llm.config.vocab_size)
        shift_constrain_mask = shift_constrain_mask.view(-1, self.model.llm.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        mask = shift_labels != -100
        num_denominators = shift_constrain_mask.sum(dim=-1)
        mask = mask & (num_denominators > 1)

        shift_labels = shift_labels[mask]
        shift_logits = shift_logits[mask]
        shift_logits_cp = shift_logits_cp[mask]
        shift_constrain_mask = shift_constrain_mask[mask]
        num_denominators = num_denominators[mask]

        legal_token_num = shift_constrain_mask.sum(dim=1)
        """To ensure computational efficiency, we calculate the mean and variance only on the samples corresponding to the first token in each sequence."""
        mask_row = legal_token_num == self.first_valid_token_num
        average_legal_token_num = self.first_valid_token_num

        pos_logits = shift_logits.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
        pos_logits = pos_logits[mask_row]
        neg_logits = shift_logits[mask_row][torch.isfinite(shift_logits[mask_row])]
        all_pos_logits = self.gather_valid_data(pos_logits)
        all_neg_logits = self.gather_valid_data(neg_logits)

        pos_mu = all_pos_logits.mean()
        neg_mu = all_neg_logits.mean()
        neg_var = all_neg_logits.var()

        diff_mu = pos_mu - neg_mu

        C = torch.log(average_legal_token_num * self.eta)
        temp_value = diff_mu**2 - 2 * C * neg_var
        temp_value = torch.clamp(temp_value, 0)
        tau = (diff_mu + torch.sqrt(temp_value)) / (2 * C)

        tau = torch.clamp(tau, self.tau_min, self.tau_max)
        tau = tau.detach()
        tau = self.ada_tau * (1 - self.alpha) + tau * self.alpha
        self.ada_tau = tau

        if self.warm_up != -1 and current_step < self.warm_up:
            tau = self.tau
            self.ada_tau = self.tau

        pos_logits = shift_logits.gather(1, shift_labels.unsqueeze(1)).squeeze(1) / tau
        pos_loss = -pos_logits
        neg_logits = torch.exp(shift_logits / tau)
        neg_loss = torch.log(neg_logits.sum(dim=-1))
        loss += tau * (pos_loss + neg_loss).mean()

        return (loss, outputs) if return_outputs else loss

class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features):
        features_copy = copy.deepcopy(features)

        constrain_mask_values = [feature["constrain_mask"] for feature in features_copy if "constrain_mask" in feature]
        his_values = [feature['his_embs'] for feature in features_copy if 'his_embs' in feature]
        candidate_values = [feature["candidates_embs"] for feature in features_copy if "candidates_embs" in feature]

        for feature in features_copy:
            feature.pop("constrain_mask")
            feature.pop("his_embs") if "his_embs" in feature else None
            feature.pop("candidates_embs") if "candidates_embs" in feature else None

        batch = super().__call__(features_copy)
        batch["constrain_mask"] = pad_sequence(
            constrain_mask_values, batch_first=True, padding_value=1, padding_side="left"
        )
        if len(his_values) != 0:
            max_his_len = max([his.shape[0] for his in his_values])
            for i in range(len(his_values)):
                ev = his_values[i]
                if ev.shape[0] < max_his_len:
                    padding = torch.zeros((max_his_len - ev.shape[0], ev.shape[1]))
                    ev = torch.cat((padding, ev), dim=0)
                his_values[i] = ev
        if len(his_values) > 0:
            batch["his_embs"] = torch.stack(his_values, dim=0)
        if len(candidate_values) > 0:
            batch["candidate_embs"] = torch.stack(candidate_values, dim=0)
        return batch


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
    ####################
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    transformers.set_seed(config.seed)
    instruction_prompt, history_prompt = get_prompt()
    print(f"Instruction Prompt: {instruction_prompt}")
    print(f"History Prompt: {history_prompt}")
    accelerator = Accelerator()

    need_data_dir = config.need_data_dir(PROJECT_ROOT)
    id2title_path = os.path.join(need_data_dir, 'entities/concepts.csv')
    columns = ['index', 'id']
    concept_df = pd.read_csv(id2title_path, header=None)
    concept_df.columns = columns
    num_items = len(concept_df)
    print('All Concepts Num: ', num_items)
    id2title = dict(zip(concept_df['index'], concept_df['id']))
    id2title_dict = {k: v.replace('K_', '') for k, v in id2title.items()}
    print(id2title_dict[1])
    vk_dict = pkl.load(open(os.path.join(need_data_dir, 'relations/vk.p'), 'rb'))
    k_embs = np.load(os.path.join(PROJECT_ROOT, config.k_emb_path))

    train_data = generate_list_from_pkl(
        data_path=os.path.join(PROJECT_ROOT, config.train_data_path),
        vk_dict=vk_dict,
        Emb_k=k_embs,
        id2title_dict=id2title_dict,
        system_prompt=instruction_prompt,
        input_prompt=history_prompt,
        num_his=config.num_his,
    )

    accelerator.wait_for_everyone()
    output_dir = config.create_run_dir(PROJECT_ROOT)

    micro_batch_size = config.batch_size
    gradient_accumulation_steps = config.batch_size // micro_batch_size

    # init  model
    model = LlaRA(model_path=config.base_model, config=config, device=config.device, is_lora=True)
    tokenizer = model.tokenizer

    # define prompt and construct train data
    assistant_sep = config.assistant_sep
    assistant_sep_token = tokenizer.encode(assistant_sep)
    end_sep = config.end_sep
    end_sep_token = tokenizer.encode(end_sep)

    # MarisaTrie
    titles_list = list(id2title_dict.values())
    tokens_list = [tokenizer.encode(f'{title}') + end_sep_token for title in titles_list]
    print(tokens_list[0])
    trie = Trie()
    for tokens in tokens_list:
        trie.insert(tokens)
    vocab_size = len(tokenizer)
    allowed_tokens_list = trie.valid_tokens(tokens_list[0])[:-1]    # because the end step in <im_end>, -1 is [] need to filter out
    print('first token predict:', len(allowed_tokens_list[0]))

    def get_candidates(uid, pid, his_concepts, num_items, neg_num=config.neg_num):
        historical_items = his_concepts
        total_negs = list(set(range(num_items)) - set(historical_items))
        candidates = random.sample(total_negs, neg_num)
        candidates.append(pid)
        random.shuffle(candidates)
        return candidates

    def tokenize(prompt):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=config.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point, Emb_k, is_warm):
        candidates = get_candidates(data_point['uid'], data_point['pid'], data_point['his_concepts'], num_items)
        candidate_embs = Emb_k[candidates].reshape(-1, Emb_k.shape[1])

        if is_warm:
            candidates_text = ', '.join([f'{id2title_dict[candidate]} <PH>' for candidate in candidates])
            my_input = data_point['input']
            warm_input = data_point['warm_input'].replace('<candidates>', candidates_text) if '<candidates>' in data_point['warm_input'] else data_point['warm_input']
            warm_user_prompt, _ = generate_prompt(data_point['instruction'], warm_input, my_input)
            warm_full_prompt = warm_user_prompt + data_point["output"] + end_sep
            tokenized_full_prompt = tokenize(warm_full_prompt)
            tokenized_user_prompt = tokenize(warm_user_prompt)

        else:
            candidates_text = ', '.join([f'({id2title_dict[candidate]} <concept_des>)' for candidate in candidates])
            my_input = data_point['input'].replace('<candidates>', candidates_text) if '<candidates>' in data_point['input'] else data_point['input']
            warm_input = data_point['warm_input']
            _, user_prompt = generate_prompt(data_point['instruction'], warm_input, my_input)
            full_prompt = user_prompt + data_point["output"] + end_sep
            tokenized_full_prompt = tokenize(full_prompt)
            tokenized_user_prompt = tokenize(user_prompt)

        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        input_ids = tokenized_full_prompt["input_ids"]
        constrain_mask = torch.ones(size=(len(input_ids) - user_prompt_len, vocab_size), dtype=torch.bool)

        response_idx_end = None
        for i in range(len(input_ids) - len(assistant_sep_token), -1, -1):
            if input_ids[i: i + len(assistant_sep_token)] == assistant_sep_token:
                response_idx_end = i + len(assistant_sep_token)
                break
        if response_idx_end is None:
            print("Response not found in input_ids")
            return None

        title_tokens = input_ids[response_idx_end:]
        allowed_tokens_list = trie.valid_tokens(title_tokens)[:-1]
        assert constrain_mask.shape[0] == len(allowed_tokens_list)

        for i, allowed_tokens in enumerate(allowed_tokens_list):
            mask = torch.zeros(vocab_size, dtype=torch.bool)
            mask[allowed_tokens] = True
            constrain_mask[i] = mask

        if not is_warm:
            tokenized_full_prompt["his_embs"] = torch.from_numpy(data_point['his_embs'])
            tokenized_full_prompt["candidates_embs"] = torch.from_numpy(candidate_embs)
        tokenized_full_prompt["constrain_mask"] = constrain_mask
        return tokenized_full_prompt

    class Prompt_dataset(Dataset):
        def __init__(self, datalist, Embk):
            self.datalist = datalist
            self.Embk = Embk
            self.max_length = len(datalist)
            self.current_index = 1

        def __len__(self):
            return len(self.datalist)

        def __getitem__(self, idx):
            data = None
            thresh_hold = self.current_index / self.max_length
            p = random.random()
            while data is None:
                if p < thresh_hold:
                    is_warm = False
                else:
                    is_warm = True
                data = generate_and_tokenize_prompt(self.datalist[idx], self.Embk, is_warm)
                idx += 1

            self.current_index += 1
            if self.current_index >= self.max_length:
                self.current_index = 1
            return data

    # test generate_and_tokenize_prompt
    train_data_0 = generate_and_tokenize_prompt(train_data[0], k_embs, False)
    first_valid_token_num = train_data_0["constrain_mask"].sum(dim=1)[0]
    train_data = Prompt_dataset(train_data, k_embs)
    assert first_valid_token_num == len(allowed_tokens_list[0])

    print('warm:', len(train_data)*config.num_epochs//20)

    data_collator = CustomDataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=config.pad_to_multiple_of)
    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        data_collator=data_collator,
        tau=config.tau,
        eta=config.eta,
        alpha=config.alpha,
        warm_up=config.warm_up,
        tau_min=config.tau_min,
        tau_max=config.tau_max,
        first_valid_token_num=first_valid_token_num,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            num_train_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            bf16=config.bf16,
            tf32=config.tf32,
            # lr_scheduler_type="cosine",
            optim=config.optim,
            logging_strategy="steps",
            logging_steps=config.logging_steps,
            save_strategy="steps",
            save_steps=config.save_steps,
            save_on_each_node=False,
            log_on_each_node=False,
            ddp_find_unused_parameters=None,
            local_rank=int(os.environ.get("LOCAL_RANK", -1)),
            seed=config.seed,
            data_seed=config.seed,
            remove_unused_columns=False,
            report_to='none'
        )
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
        model, data_collator([train_data_0]),
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


# 多组实验：每项是 Config 的覆盖字段，按顺序训练；每个 run 按 config md5 落在独立目录
SWEEP = [
    # {
    #     'base_model': '/home/enerpat/weiliting/local_llm/Qwen2.5-1.5B',
    #     'seed': 12,
    #     'k_emb_path': '/home/enerpat/weiliting/llm_rec/sasrec_new/runs/small_data/prepare_data/39ca25ef14821014de76d4dbb3ead1b8/Emb_k.npy',
    #     'train_data_path': '/home/enerpat/weiliting/llm_rec/prepare_data/train_data_new.p',
    # },
    # {
    #     'base_model': '/home/enerpat/weiliting/local_llm/Qwen2.5-1.5B',
    #     'seed': 22,
    #     'k_emb_path': '/home/enerpat/weiliting/llm_rec/sasrec_new/runs/small_data/prepare_data/5b57dda4ac547bc823e7201d1ba5f184/Emb_k.npy',
    #     'train_data_path': '/home/enerpat/weiliting/llm_rec/prepare_data/train_data_new.p',
    # },
    # {
    #     'base_model': '/home/enerpat/weiliting/local_llm/Qwen2.5-1.5B',
    #     'seed': 32,
    #     'k_emb_path': '/home/enerpat/weiliting/llm_rec/sasrec_new/runs/small_data/prepare_data/a8d002ef9b217cff7e95b19f6564713b/Emb_k.npy',
    #     'train_data_path': '/home/enerpat/weiliting/llm_rec/prepare_data/train_data_new.p',
    # },
    # {
    #     'base_model': '/home/enerpat/weiliting/local_llm/Qwen2.5-1.5B',
    #     'seed': 52,
    #     'k_emb_path': '/home/enerpat/weiliting/llm_rec/sasrec_new/runs/small_data/prepare_data/9c6490ad686d811a0ca357becd22ab48/Emb_k.npy',
    #     'train_data_path': '/home/enerpat/weiliting/llm_rec/prepare_data/train_data_new.p',
    # },
    # {
    #     'dataset_dir': 'data/MOOCCube/mid_data',
    #     'base_model': '/root/weiliting/local_llm/Qwen2.5-1.5B',
    #     'seed': 12,
    #     'k_emb_path': '/root/weiliting/llm_rec/sasrec_new/runs/mid_data/prepare_data/ad7ec1abda0fd5b2a0d6e51aacf6b2b3/Emb_k.npy',
    #     'train_data_path': '/root/weiliting/llm_rec/prepare_data/mooc_sasrec_augllm2/train_data_new.p',
    # },
    # {
    #     'dataset_dir': 'data/MOOCCube/mid_data',
    #     'base_model': '/root/weiliting/local_llm/Qwen2.5-1.5B',
    #     'seed': 22,
    #     'k_emb_path': '/root/weiliting/llm_rec/sasrec_new/runs/mid_data/prepare_data/eb4baec60d7c628ca45ff910b1279f28/Emb_k.npy',
    #     'train_data_path': '/root/weiliting/llm_rec/prepare_data/mooc_sasrec_augllm2/train_data_new.p',
    # },
    {
        'dataset_dir': 'data/MOOCCube/mid_data',
        'base_model': '../../local_llm/Qwen2.5-1.5B',
        'seed': 32,
        'k_emb_path': '../sasrec_new/runs/mid_data/prepare_data/5b1cac37bbb0d5f86f3771a9f25338df/Emb_k.npy',
        'train_data_path': '../prepare_data/mooc_sasrec_augllm2/train_data_new.p',
    },
    {
        'dataset_dir': 'data/MOOCCube/mid_data',
        'base_model': '../../local_llm/Qwen2.5-1.5B',
        'seed': 52,
        'k_emb_path': '../sasrec_new/runs/mid_data/prepare_data/327b92b578defd2f295a6714f83b0c60/Emb_k.npy',
        'train_data_path': '../prepare_data/mooc_sasrec_augllm2/train_data_new.p',
    },
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
    """单次运行：python train_mooc_base.py run --tau 2.0 --alpha 0.1"""
    train(Config(**_cast(overrides)))


def sweep():
    """跑 SWEEP 里的全部配置：python train_mooc_base.py sweep"""
    for i, overrides in enumerate(SWEEP):
        print(f'===== [{i + 1}/{len(SWEEP)}] {overrides or "default"} =====')
        train(Config(**overrides))
        torch.cuda.empty_cache()


if __name__ == '__main__':
    fire.Fire({'train': train, 'run': run, 'sweep': sweep})