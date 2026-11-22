"""ALLMRec 历史链路（语义对齐旧工程 ALLMRec/train_mooc_history.py）。
用法：
    python ALLMRec/train_mooc_history.py train
    python ALLMRec/train_mooc_history.py run --num_epochs 5 --batch_size 16
    python ALLMRec/train_mooc_history.py sweep
"""

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
import pandas as pd
import torch
import transformers
from accelerate import Accelerator
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, PROJECT_ROOT
from pt_reasoner import PTReasoner
from Trie import Trie


def get_prompt(dataset_name):
    instruction_prompt, history_prompt = None, None
    if "MOOC" in dataset_name:
        instruction_prompt = "The description of user is: <user_des>. Given a list of concepts the user has learned before, please recommend a new concept from the candidates that the user likes to the user."
        history_prompt = "The user has learned the following concepts before: {historical_concepts}\nThe candidates you need to select: {candidates}\nPlease return your choice."

    return instruction_prompt, history_prompt


def generate_list_from_pkl(train_data_path, vk_dict, Emb_u, Emb_k, id2title_dict, instruction_str, input_prefix_str, num_his=20):
    train_data = pkl.load(open(train_data_path, 'rb'))
    json_data = []

    for index, data in enumerate(train_data):
        his_concepts = []
        for v in data[1]:
            his_concepts.extend(vk_dict[v])
        his_concepts = list(set(his_concepts))

        user_emb = Emb_u[index].reshape(1, -1)
        his_videos = data[1][-10:]
        current_concepts = []
        for v in his_videos:
            current_concepts.extend(vk_dict[v])
        current_concepts = current_concepts[-num_his:]

        his_embs = Emb_k[current_concepts].reshape(-1, Emb_k.shape[1])
        label = data[2]

        input_str = input_prefix_str.format(**{'historical_concepts': ','.join([f'{i+1}.({id2title_dict[current_concepts[i]]}, <his_des>)' for i in range(len(current_concepts))]), 'candidates': '<candidates>'}) \
            if 'candidates' in input_prefix_str else input_prefix_str.format(**{'historical_concepts': ','.join([f'{i+1}.({id2title_dict[current_concepts[i]]}, <his_des>)' for i in range(len(current_concepts))])})
        label_str = f'{id2title_dict[label]}'

        json_entry = {'instruction': instruction_str, 'input': f'{input_str}', 'output': f'{label_str}', 'uid': data[0], 'pid': label, 'his_concepts': his_concepts, 'his_embs': his_embs, 'user_emb': user_emb}
        json_data.append(json_entry)
    return json_data


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable params: {trainable_params} || All params: {all_param} || Trainable %: {100 * trainable_params / all_param:.1f}%")


def generate_prompt(instruction, input):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


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

        loss = None
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
        loss = tau * (pos_loss + neg_loss).mean()

        return (loss, outputs) if return_outputs else loss


class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features):
        features_copy = copy.deepcopy(features)

        constrain_mask_values = [feature["constrain_mask"] for feature in features_copy]
        user_values = [feature["user_emb"] for feature in features_copy]
        video_values = [feature['his_embs'] for feature in features_copy]
        candidate_values = [feature["candidates_embs"] for feature in features_copy]

        for feature in features_copy:
            feature.pop("constrain_mask")
            feature.pop("user_emb")
            feature.pop("his_embs")
            feature.pop("candidates_embs")

        batch = super().__call__(features_copy)
        batch["constrain_mask"] = pad_sequence(
            constrain_mask_values, batch_first=True, padding_value=1, padding_side="left"
        )
        max_video_len = max([video.shape[0] for video in video_values])
        for i in range(len(video_values)):
            ev = video_values[i]
            if ev.shape[0] < max_video_len:
                padding = torch.zeros((max_video_len - ev.shape[0], ev.shape[1]))
                ev = torch.cat((padding, ev), dim=0)
            video_values[i] = ev

        batch["user_emb"] = torch.stack(user_values, dim=0)
        batch["his_embs"] = torch.stack(video_values, dim=0)
        batch["candidate_embs"] = torch.stack(candidate_values, dim=0)
        return batch


def _gpu_peak_gb():
    return torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0


def _measure_infer_latency(model, batch, repeats=20, max_new_tokens=16, warmup=3):
    """单用户（batch=1）端到端推理时延：QMoE 拼 embedding + 自回归生成，返回 ms/user。"""
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
    #########
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    transformers.set_seed(config.seed)
    # dataset_dir 形如 data/MOOCCube/small_data，含 'MOOC'
    instruction_prompt, history_prompt = get_prompt(config.dataset_dir)
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
    u_embs = np.load(os.path.join(PROJECT_ROOT, config.u_emb_path))

    train_data = generate_list_from_pkl(
        train_data_path=os.path.join(PROJECT_ROOT, config.train_data_path),
        vk_dict=vk_dict,
        Emb_u=u_embs,
        Emb_k=k_embs,
        id2title_dict=id2title_dict,
        instruction_str=instruction_prompt,
        input_prefix_str=history_prompt,
        num_his=config.num_his,
    )

    accelerator.wait_for_everyone()
    output_dir = config.create_run_dir(PROJECT_ROOT)

    micro_batch_size = config.micro_batch_size or config.batch_size
    gradient_accumulation_steps = config.batch_size // micro_batch_size

    # init  model
    model = PTReasoner(model_path=config.base_model, config=config, device=config.device, is_lora=True)
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

    def get_candidates(uid, pid, his_concepts, num_items, is_train, neg_num=config.neg_num):
        historical_items = his_concepts
        total_negs = list(set(range(num_items)) - set(historical_items))
        candidates = random.sample(total_negs, neg_num)
        candidates.append(pid)
        if is_train:
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

    def generate_and_tokenize_prompt(data_point, Emb_k, is_train):
        candidates = get_candidates(data_point['uid'], data_point['pid'], data_point['his_concepts'], num_items, is_train)
        candidate_embs = Emb_k[candidates].reshape(-1, Emb_k.shape[1])

        candidates_text = ', '.join([f'({id2title_dict[candidate]}, <concept_des>)' for candidate in candidates])
        my_input = data_point['input'].replace('<candidates>', candidates_text) if '<candidates>' in data_point['input'] else data_point['input']
        user_prompt = generate_prompt(data_point['instruction'], my_input)
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
        tokenized_full_prompt["constrain_mask"] = constrain_mask

        tokenized_full_prompt["user_emb"] = torch.from_numpy(data_point['user_emb'])
        tokenized_full_prompt["his_embs"] = torch.from_numpy(data_point['his_embs'])
        tokenized_full_prompt["candidates_embs"] = torch.from_numpy(candidate_embs)
        return tokenized_full_prompt

    class Prompt_dataset(Dataset):
        def __init__(self, datalist, Embk, is_train):
            self.datalist = datalist
            self.Embk = Embk
            self.is_train = is_train

        def __len__(self):
            return len(self.datalist)

        def __getitem__(self, idx):
            data = None
            while data is None:
                data = generate_and_tokenize_prompt(self.datalist[idx], self.Embk, self.is_train)
                idx += 1
            return data

    # test generate_and_tokenize_prompt
    train_data_0 = generate_and_tokenize_prompt(train_data[0], k_embs, True)
    first_valid_token_num = train_data_0["constrain_mask"].sum(dim=1)[0]
    train_data = Prompt_dataset(train_data, k_embs, True)
    assert first_valid_token_num == len(allowed_tokens_list[0])

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
            report_to='none',
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
    {
        'dataset_dir': 'data/MOOCCube/mid_data',
        'base_model': '/home/root/local_llm/Qwen2.5-1.5B',
        'seed': 12,
        'k_emb_path': '/home/root/llm_rec/sasrec_new/runs/mid_data/prepare_data/ad7ec1abda0fd5b2a0d6e51aacf6b2b3/Emb_k.npy',
        'u_emb_path': '/home/root/llm_rec/sasrec_new/runs/mid_data/prepare_data/ad7ec1abda0fd5b2a0d6e51aacf6b2b3/train_user_features.npy',
        'train_data_path': '/home/root/llm_rec/prepare_data/mooc_sasrec_augllm2/train_data_new.p',
        'save_root': 'ALLMRec_new/runs'
    },
    {
        'dataset_dir': 'data/MOOCCube/mid_data',
        'base_model': '/home/root/local_llm/Qwen2.5-1.5B',
        'seed': 22,
        'k_emb_path': '/home/root/llm_rec/sasrec_new/runs/mid_data/prepare_data/eb4baec60d7c628ca45ff910b1279f28/Emb_k.npy',
        'u_emb_path': '/home/root/llm_rec/sasrec_new/runs/mid_data/prepare_data/eb4baec60d7c628ca45ff910b1279f28/train_user_features.npy',
        'train_data_path': '/home/root/llm_rec/prepare_data/mooc_sasrec_augllm2/train_data_new.p',
        'save_root': 'ALLMRec_new/runs'
    },
    {
        'dataset_dir': 'data/MOOCCube/mid_data',
        'base_model': '/home/root/local_llm/Qwen2.5-1.5B',
        'seed': 32,
        'k_emb_path': '/home/root/llm_rec/sasrec_new/runs/mid_data/prepare_data/5b1cac37bbb0d5f86f3771a9f25338df/Emb_k.npy',
        'u_emb_path': '/home/root/llm_rec/sasrec_new/runs/mid_data/prepare_data/5b1cac37bbb0d5f86f3771a9f25338df/train_user_features.npy',
        'train_data_path': '/home/root/llm_rec/prepare_data/mooc_sasrec_augllm2/train_data_new.p',
        'save_root': 'ALLMRec_new/runs'
    },
    {
        'dataset_dir': 'data/MOOCCube/mid_data',
        'base_model': '/home/root/local_llm/Qwen2.5-1.5B',
        'seed': 52,
        'k_emb_path': '/home/root/llm_rec/sasrec_new/runs/mid_data/prepare_data/327b92b578defd2f295a6714f83b0c60/Emb_k.npy',
        'u_emb_path': '/home/root/llm_rec/sasrec_new/runs/mid_data/prepare_data/327b92b578defd2f295a6714f83b0c60/train_user_features.npy',
        'train_data_path': '/home/root/llm_rec/prepare_data/mooc_sasrec_augllm2/train_data_new.p',
        'save_root': 'ALLMRec_new/runs'
    },


    # {
    #     'dataset_dir': 'data/MOOCCube/small_data',
    #     'base_model': '/home/root/local_llm/Qwen2.5-1.5B',
    #     'seed': 12,
    #     'k_emb_path': '/home/root/llm_rec/sasrec_new/runs/small_data/prepare_data/39ca25ef14821014de76d4dbb3ead1b8/Emb_k.npy',
    #     'u_emb_path': '/home/root/llm_rec/sasrec_new/runs/small_data/prepare_data/39ca25ef14821014de76d4dbb3ead1b8/train_user_features.npy',
    #     'train_data_path': '/home/root/llm_rec/prepare_data/train_data_new.p',
    #     'save_root': 'ALLMRec_new/runs'
    # },
    # {
    #     'dataset_dir': 'data/MOOCCube/small_data',
    #     'base_model': '/home/root/local_llm/Qwen2.5-1.5B',
    #     'seed': 22,
    #     'k_emb_path': '/home/root/llm_rec/sasrec_new/runs/small_data/prepare_data/5b57dda4ac547bc823e7201d1ba5f184/Emb_k.npy',
    #     'u_emb_path': '/home/root/llm_rec/sasrec_new/runs/small_data/prepare_data/5b57dda4ac547bc823e7201d1ba5f184/train_user_features.npy',
    #     'train_data_path': '/home/root/llm_rec/prepare_data/train_data_new.p',
    #     'save_root': 'ALLMRec_new/runs'
    # },
    # {
    #     'dataset_dir': 'data/MOOCCube/small_data',
    #     'base_model': '/home/root/local_llm/Qwen2.5-1.5B',
    #     'seed': 32,
    #     'k_emb_path': '/home/root/llm_rec/sasrec_new/runs/small_data/prepare_data/a8d002ef9b217cff7e95b19f6564713b/Emb_k.npy',
    #     'u_emb_path': '/home/root/llm_rec/sasrec_new/runs/small_data/prepare_data/a8d002ef9b217cff7e95b19f6564713b/train_user_features.npy',
    #     'train_data_path': '/home/root/llm_rec/prepare_data/train_data_new.p',
    #     'save_root': 'ALLMRec_new/runs'
    # },
    # {
    #     'dataset_dir': 'data/MOOCCube/small_data',
    #     'base_model': '/home/root/local_llm/Qwen2.5-1.5B',
    #     'seed': 52,
    #     'k_emb_path': '/home/root/llm_rec/sasrec_new/runs/small_data/prepare_data/9c6490ad686d811a0ca357becd22ab48/Emb_k.npy',
    #     'u_emb_path': '/home/root/llm_rec/sasrec_new/runs/small_data/prepare_data/9c6490ad686d811a0ca357becd22ab48/train_user_features.npy',
    #     'train_data_path': '/home/root/llm_rec/prepare_data/train_data_new.p',
    #     'save_root': 'ALLMRec_new/runs'
    # },
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
    """单次运行：python ALLMRec/train_mooc_history.py run --tau 2.0 --alpha 0.1"""
    train(Config(**_cast(overrides)))


def sweep():
    """跑 SWEEP 里的全部配置：python ALLMRec/train_mooc_history.py sweep"""
    for i, overrides in enumerate(SWEEP):
        print(f'===== [{i + 1}/{len(SWEEP)}] {overrides or "default"} =====')
        train(Config(**overrides))
        torch.cuda.empty_cache()


if __name__ == '__main__':
    fire.Fire({'train': train, 'run': run, 'sweep': sweep})
