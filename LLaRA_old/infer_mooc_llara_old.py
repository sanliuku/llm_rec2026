"""LLaRA_old 推理：预测部分照 LLaRA 原链路的 predict 脚本，只换输入。















"""

import json
import os
import pickle as pkl
import sys

import fire
import numpy as np
import pandas as pd
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import gather_object
from genre.trie import MarisaTrie
from peft import PeftModel
from pydantic import BaseModel
from tqdm import tqdm
from transformers import GenerationConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt import get_prompt
from reasoner import LlaRA

# ------------------------------------------------------------------ 写死的路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_MODEL = '../../local_llm/Qwen2.5-1.5B'
# 训练完把 run_id 填这里（目录里要有 adapter_config.json 和 concept_moe_adaptor.pth）
LORA_DIR = os.path.join(ROOT, 'LLaRA_old/runs/mid_data/results/69a705cd45cb25196cdc9e95fcf6cd85')

CONCEPT_CSV = os.path.join(ROOT, 'data/MOOCCube/mid_data/need_data/entities/concepts.csv')
# 与 LLaRA_old/config.py 的 k_emb_path 一致（sasrec_old 的 CF 物品向量，256 维）
K_EMB_PATH = os.path.join(ROOT, 'sasrec_old/runs/mid_data/prepare_data/116719ce0d1bed7972d4defb6130686c/Emb_k.npy')
TEST_DATA = os.path.join(ROOT, 'prepare_data/mooc_sasrec_augllm2/test_concept_data.p')

# 候选集：两种口径二选一
CAND_SOURCE = 'sasrec_global'
CAND_PKL = os.path.join(ROOT, 'sasrec_old/runs/mid_data/results/116719ce0d1bed7972d4defb6130686c/top20_best_sasrec_old_test_global.pkl')

NUM_HIS = 20
MAX_NEW_TOKENS = 128


class Config(BaseModel):
    lora_weights_path: str = LORA_DIR
    dataset: str = 'MOOC'
    embed_size: int = 256
    emb_dim: int = embed_size
    token_size: int = 1536
    hidden_size: int = token_size
    cutoff_len: int = 1024

    batch_size: int = 8
    base_model: str = BASE_MODEL
    num_beams: int = 10
    constraint_BS: int = 1
    seed: int = 42
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = ['k_proj', 'v_proj', 'q_proj', 'o_proj',
                                 'gate_proj', 'up_proj', 'down_proj']
    device: str = 'cuda:0'


def _norm(text: str) -> str:
    return ' '.join(str(text).split()).lower()


def build_items(source: str, cand_path: str, rows, num_his: int = NUM_HIS) -> list:
    """候选集 → [(row, uid, his, cands, answer)]，id 统一成 1-based。

    align_masked 是 per-row（53911 条，行序对齐 concept_data），
    sasrec_global 是 per-user（1000 条，一个用户一个 query）。
    """
    data = pkl.load(open(cand_path, 'rb'))

    if source == 'align_masked':
        cands = [[int(c) + 1 for c in x] for x in data['topk_items']]
        targets = [int(t) + 1 for t in data['truth_labels']]
        if len(targets) != len(rows):
            raise ValueError(f'候选集 {len(targets)} 行，concept_data {len(rows)} 行，对不上')
        bad = [i for i in range(len(rows)) if int(rows[i][2]) != targets[i]]
        if bad:
            raise ValueError(f'{len(bad)} 行 label 与 target 不一致（首例 row={bad[0]}）')
        return [{'row': i, 'uid': int(r[0]), 'his': [int(x) for x in r[1] if x][-num_his:],
                 'cands': cands[i], 'answer': {targets[i]}}
                for i, r in enumerate(rows)]

    if source != 'sasrec_global':
        raise ValueError(f"CAND_SOURCE 只能是 'align_masked'/'sasrec_global'，收到 {source!r}")

    first_row, his_by_user = {}, {}
    for i, r in enumerate(rows):
        u = int(r[0])
        if u not in first_row:
            first_row[u] = i
            his_by_user[u] = [int(x) for x in r[1] if x][-num_his:]

    topk, truth_by_user = data['topk_items'], data['truth_by_user']
    return [{'row': first_row[u], 'uid': u, 'his': his_by_user[u],
             'cands': [int(c) + 1 for c in topk[u]],
             'answer': {int(v) + 1 for v in truth_by_user[u]}}
            for u in range(len(topk))]


def generate_list_from_pkl(items, k_embs, id2title_dict, system_prompt, input_prompt):
    """拼 prompt 和两类向量（his_embs / candidate_embs），一个 concept 后面跟一个占位符。"""
    json_data = []
    for it in items:
        his_ids, cand_ids = it['his'], it['cands']

        seq_t2 = ','.join(f'{i + 1}.{id2title_dict[c - 1]} <his_des>'
                          for i, c in enumerate(his_ids))
        input_str = input_prompt.format(historical_concepts=seq_t2, candidates='<candidates>')
        candidates_text = ', '.join(f'({id2title_dict[c - 1]} <concept_des>)' for c in cand_ids)
        input_str = input_str.replace('<candidates>', candidates_text)

        answer = [id2title_dict[c - 1] for c in sorted(it['answer'])]
        json_data.append({
            'uid': it['uid'], 'row': it['row'],
            'instruction': system_prompt,
            'input': input_str,
            'output': answer[0],
            'answer': answer,
            'candidate_list': [id2title_dict[c - 1] for c in cand_ids],
            'his_embs': k_embs[[c - 1 for c in his_ids]].reshape(-1, k_embs.shape[1]),
            'candidate_embs': k_embs[[c - 1 for c in cand_ids]].reshape(-1, k_embs.shape[1]),
        })
    return json_data


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Response:
"""


def first_hit_rank(beams, answers) -> list:
    """第一个命中答案的 beam 名次（0-based）；都没命中返回 -1（answers 已归一化）。"""
    for i, text in enumerate(beams):
        got = _norm(text)
        for a in answers:
            if got == a or (len(got) > 2 and a in got):
                return i
    return -1


def compute_metrics(ranks, ks=(1, 5, 10)) -> dict:
    total = len(ranks)
    out = {'n': total, 'hit_rate': sum(1 for r in ranks if r >= 0) / total if total else 0.0}
    for k in ks:
        out[f'hr@{k}'] = sum(1 for r in ranks if 0 <= r < k) / total if total else 0.0
        out[f'ndcg@{k}'] = sum(1.0 / np.log2(r + 2) for r in ranks if 0 <= r < k) / total if total else 0.0
    return out


def load_infer_model(config: Config, device: str) -> LlaRA:
    """8bit 底模 + LoRA 权重 + concept_moe_adaptor；不做 merge / torch.compile。"""
    model = LlaRA(model_path=config.base_model, config=config, device=device, is_lora=True)
    # model.llm = PeftModel.from_pretrained(model.llm, config.lora_weights_path, is_trainable=False)
    # model.embedding_converter_concept.load_state_dict(
    #     torch.load(os.path.join(config.lora_weights_path, 'concept_moe_adaptor.pth'),
    #                map_location=device))
    # model.embedding_converter_concept.to(device)
    model.load_model(config)  
    model.eval()
    return model


def main():
    config = Config()
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    transformers.set_seed(config.seed)

    accelerator = Accelerator()
    accelerator.print('LoRA Weights Path: ', config.lora_weights_path)
    accelerator.print('Candidates: ', CAND_SOURCE, CAND_PKL)

    instruction_prompt, history_prompt = get_prompt()
    concept_df = pd.read_csv(CONCEPT_CSV, header=None, names=['index', 'id'])
    accelerator.print('All Concepts Num: ', len(concept_df))
    id2title = dict(zip(concept_df['index'], concept_df['id']))
    id2title_dict = {k: v.replace('K_', '') for k, v in id2title.items()}

    k_embs = np.load(K_EMB_PATH)
    test_data = generate_list_from_pkl(
        items=build_items(CAND_SOURCE, CAND_PKL, pkl.load(open(TEST_DATA, 'rb'))),
        k_embs=k_embs,
        id2title_dict=id2title_dict,
        system_prompt=instruction_prompt,
        input_prompt=history_prompt,
    )
    accelerator.print(f'test samples: {len(test_data)}')

    result_json_data = os.path.join(config.lora_weights_path,
                                    f'predict_{CAND_SOURCE}.json')
    if os.path.exists(result_json_data):
        accelerator.print(f'The {result_json_data} has existed.')
        return
    accelerator.wait_for_everyone()

    model = load_infer_model(config, config.device)
    tokenizer = model.tokenizer
    for _, param in model.named_parameters():
        param.requires_grad = False

    assistant_sep = '### Response:\n'
    assistant_sep_token = tokenizer.encode(assistant_sep)
    end_sep = '<|endoftext|>'
    end_sep_token = tokenizer.encode(end_sep)

    def create_custom_prefix_fn(candidate_lists, tokenizer):
        batch2trie = {}
        for index in range(len(candidate_lists)):
            candidate_list = candidate_lists[index]
            tokens_list = [tokenizer.encode(assistant_sep + f'{title}') for title in candidate_list]
            trie = MarisaTrie(tokens_list)
            batch2trie[index] = trie

        def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> list:
            input_ids = input_ids.tolist()
            for i in range(len(input_ids)):
                if input_ids[i: i + len(assistant_sep_token)] == assistant_sep_token:
                    break

            prefix = input_ids[i:]
            trie = batch2trie[batch_id]
            allowed_tokens = trie.get(prefix)
            allowed_tokens = [end_sep_token] if allowed_tokens == [] else allowed_tokens

            return allowed_tokens

        return prefix_allowed_tokens_fn

    def evaluate(instructions, inputs, candidate_lists, his_emb_list, candidate_emb_list,
                 num_beams=config.num_beams, max_new_tokens=MAX_NEW_TOKENS):
        prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True,
                           max_length=config.cutoff_len).to(accelerator.device)

        max_video_len = max([video.shape[0] for video in his_emb_list])
        for i in range(len(his_emb_list)):
            ev = his_emb_list[i]
            if ev.shape[0] < max_video_len:
                padding = np.zeros((max_video_len - ev.shape[0], ev.shape[1]))
                ev = np.concatenate([padding, ev], axis=0)
            his_emb_list[i] = ev

        his_embs = np.stack(his_emb_list, axis=0)
        inputs['his_embs'] = torch.from_numpy(his_embs).to(accelerator.device)
        candidate_embs = np.stack(candidate_emb_list, axis=0)
        inputs['candidate_embs'] = torch.from_numpy(candidate_embs).to(accelerator.device)

        input_embs = model.get_embs(inputs)
        attention_mask = inputs['attention_mask']

        custom_prefix_fn = create_custom_prefix_fn(candidate_lists, tokenizer)

        with torch.no_grad():
            generation_config = GenerationConfig(
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )
            generation_output = model.llm.generate(
                inputs_embeds=input_embs,
                attention_mask=attention_mask,
                input_ids=inputs.input_ids,
                generation_config=generation_config,
                prefix_allowed_tokens_fn=custom_prefix_fn if config.constraint_BS else None,
            )
            sequences_scores = generation_output.sequences_scores.tolist()
            sequences_scores = [
                sequences_scores[i * num_beams: (i + 1) * num_beams] for i in
                range(len(sequences_scores) // num_beams)
            ]

            output = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=False)
            output = [_.split(assistant_sep)[-1].split(end_sep)[0] for _ in output]
            real_outputs = [output[i * num_beams: (i + 1) * num_beams]
                            for i in range(len(output) // num_beams)]
        return real_outputs, sequences_scores

    def batch(list, batch_size=config.batch_size):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i: batch_size * (i + 1)]

    input_dict = {
        'instructions': [_['instruction'] for _ in test_data],
        'inputs': [_['input'] for _ in test_data],
        'candidate_lists': [_['candidate_list'] for _ in test_data],
        'his_emb_list': [_['his_embs'] for _ in test_data],
        'candidate_emb_list': [_['candidate_embs'] for _ in test_data],
    }

    with accelerator.split_between_processes(input_dict) as input_temp:
        outputs, sequences_scores = [], []

        for batch1 in tqdm(
                zip(batch(input_temp['instructions']), batch(input_temp['inputs']),
                    batch(input_temp['candidate_lists']), batch(input_temp['his_emb_list']),
                    batch(input_temp['candidate_emb_list'])),
                total=(len(input_temp['instructions']) + config.batch_size - 1) // config.batch_size,
        ):
            instructions, inputs, candidate_lists, his_emb_list, candidate_emb_list = batch1
            output, sequences_score = evaluate(instructions, inputs, candidate_lists,
                                               his_emb_list, candidate_emb_list)
            outputs.extend(output)
            sequences_scores.extend(sequences_score)

    outputs = gather_object(outputs)
    sequences_scores = gather_object(sequences_scores)
    assert len(outputs) == len(test_data)
    assert len(sequences_scores) == len(test_data)

    if accelerator.is_main_process:
        ranks = []
        for i, _ in enumerate(test_data):
            answers = {_norm(a) for a in test_data[i]['answer']}
            ranks.append(first_hit_rank(outputs[i], answers))

        for i, _ in enumerate(test_data):
            test_data[i]['predict'] = outputs[i]
            test_data[i]['scores'] = sequences_scores[i]
            test_data[i]['rank'] = ranks[i]
            test_data[i]['his_embs'] = None
            test_data[i]['candidate_embs'] = None

        with open(result_json_data, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)

        metrics = compute_metrics(ranks)
        metrics['miss'] = sum(1 for r in ranks if r < 0) / max(len(ranks), 1)
        with open(os.path.join(config.lora_weights_path, f'metrics_{CAND_SOURCE}.json'),
                  'w', encoding='utf-8') as f:
            json.dump({'cand_source': CAND_SOURCE, 'cand_path': CAND_PKL,
                       'num_beams': config.num_beams, 'constraint_BS': config.constraint_BS,
                       'metrics': metrics}, f, ensure_ascii=False, indent=2)
        print(f"[{CAND_SOURCE}] " + '  '.join(
            f'{k}={v:.5f}' if isinstance(v, float) else f'{k}={v}' for k, v in metrics.items()))
        print(f'saved: {result_json_data}')


if __name__ == '__main__':
    fire.Fire(main)
