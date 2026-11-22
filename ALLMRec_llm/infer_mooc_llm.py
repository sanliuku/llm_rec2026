"""ALLMRec_llm 推理：预测部分照原链路的 predict 脚本，只换输入。


















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
from pydantic import BaseModel
from tqdm import tqdm
from transformers import GenerationConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt import format_candidates, format_history, generate_prompt, get_prompt
from pt_reasoner import PTReasoner

# ------------------------------------------------------------------ 写死的路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_MODEL = '../../local_llm/Qwen2.5-1.5B'
# 第二阶段训练产出的 run 目录（里面有 user_adaptor.pth / concept_adaptor.pth），训练完把 <RUN_ID> 填这里
ADAPTOR_DIR = os.path.join(ROOT, 'ALLMRec_llm/runs/small_data/results/f869b292db1d3073d7ff2c1c92ac24fe')

CONCEPT_CSV = os.path.join(ROOT, 'data/MOOCCube/small_data/need_data/entities/concepts.csv')
# 第一阶段（ALLMRec_old）的产物：e_i 128 维、x^u 256 维，与 ALLMRec_llm/config.py 同一目录
STAGE1_DIR = os.path.join(ROOT, 'ALLMRec_old/runs/small_data/prepare_data/3cd4865a871d47fa9f4c521b1d4f9443')
K_EMB_PATH = os.path.join(STAGE1_DIR, 'Emb_k.npy')
U_EMB_PATH = os.path.join(STAGE1_DIR, 'test_user_features.npy')
TEST_DATA = os.path.join(ROOT, 'prepare_data/mooc_sasrec_augllm/test_concept_data.p')

# 候选集：两种口径二选一
# CAND_SOURCE = 'align_masked'
# CAND_PKL = os.path.join(
#     ROOT, 'ALLMRec_old/runs/small_data/results/3cd4865a871d47fa9f4c521b1d4f9443/'
        #   'top20_best_align_test_masked.pkl')
CAND_SOURCE = 'sasrec_global'
CAND_PKL = os.path.join(ROOT, 'ALLMRec_old/runs/small_data/results/3cd4865a871d47fa9f4c521b1d4f9443/top20_best_align_test_global.pkl')

# mid_data 口径（本机实测过的那套，本机没有 small_data 的 stage-1 产物时用这组）：
# STAGE1_DIR = os.path.join(ROOT, 'ALLMRec_old/runs/mid_data/prepare_data/877d74f25f2e7384ab16648ec9b3c997')
# U_EMB_PATH = os.path.join(STAGE1_DIR, 'test_user_features.npy')   # (53911, 256)
# TEST_DATA  = os.path.join(ROOT, 'prepare_data/mooc_sasrec_augllm2/test_concept_data.p')
# CAND_PKL   = os.path.join(ROOT, 'ALLMRec_old/runs/mid_data/results/877d74f25f2e7384ab16648ec9b3c997/'
#                                 'top20_best_align_test_masked.pkl')

NUM_HIS = 20
MAX_NEW_TOKENS = 128
MAX_SAMPLES = 0        # >0 时只跑前 N 条（冒烟用）
CONSTRAINT_BS = 1      # 1 = 候选 title 前缀约束；0 = 官方 A-LLMRec 的自由生成


class Config(BaseModel):
    lora_weights_path: str = ADAPTOR_DIR
    dataset: str = 'MOOC'
    # 两侧输入维度不同：用户是 CF 的 x^u(256)，物品是第一阶段的联合嵌入 e_i(128)
    user_embed_size: int = 256
    item_embed_size: int = 128
    token_size: int = 1536
    hidden_size: int = token_size
    user_act: str = 'leakyrelu'
    item_act: str = 'gelu'
    cutoff_len: int = 4096

    batch_size: int = 8
    base_model: str = BASE_MODEL
    num_beams: int = 10
    constraint_BS: int = CONSTRAINT_BS
    seed: int = 42
    # 官方 A-LLMRec 不训 LoRA，这几个字段只为 PTReasoner(is_lora=True) 的签名完整
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = ['q_proj', 'v_proj']
    device: str = 'cuda:0'


def _norm(text: str) -> str:
    return ' '.join(str(text).split()).lower()


def build_items(source: str, cand_path: str, rows, num_his: int = NUM_HIS) -> list:
    """候选集 → [(row, uid, his, cands, answer)]，id 统一成 1-based。

    align_masked 是 per-row（每行 concept_data 一条，行序对齐），
    sasrec_global 是 per-user（一个用户一个 query，1000 条）。
    row 用来取 x^u（{split}_user_features.npy 与 concept_data 行号一一对应）。
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


def generate_list_from_pkl(items, k_embs, u_embs, id2title_dict, system_prompt, input_prompt):
    """拼 prompt 和三路向量（user_emb / his_embs / candidate_embs）。

    prompt 里每个历史 concept 后面跟一个 <his_des>、每个候选后面跟一个 <concept_des>，
    get_embs 按这两个占位符的位置把向量填进去；<user_des> 在 instruction 里出现一次。
    """
    json_data = []
    for it in items:
        his_ids, cand_ids = it['his'], it['cands']

        input_str = input_prompt.format(
            historical_concepts=format_history(his_ids, id2title_dict),
            candidates=format_candidates(cand_ids, id2title_dict),
        )

        answer = [id2title_dict[c - 1] for c in sorted(it['answer'])]
        json_data.append({
            'uid': it['uid'], 'row': it['row'],
            'instruction': system_prompt,
            'input': input_str,
            'output': answer[0],
            'answer': answer,
            'candidate_list': [id2title_dict[c - 1] for c in cand_ids],
            'user_emb': u_embs[it['row']].astype(np.float32),
            'his_embs': k_embs[[c - 1 for c in his_ids]].reshape(-1, k_embs.shape[1]),
            'candidate_embs': k_embs[[c - 1 for c in cand_ids]].reshape(-1, k_embs.shape[1]),
        })
    return json_data


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


def load_infer_model(config: Config, device: str) -> PTReasoner:
    """8bit 底模 + 两个投影层；不做 merge / torch.compile。"""
    for name in ('user_adaptor.pth', 'concept_adaptor.pth'):
        path = os.path.join(config.lora_weights_path, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f'第二阶段权重缺失: {path}')
    model = PTReasoner(model_path=config.base_model, config=config, device=device,
                       is_lora=False, gradient_checkpointing=False)
    model.load_model(config)  # is_lora=False 时只加载两个投影层
    model.llm.config.use_cache = False  # 推理不做反向，恢复 KV cache（只影响速度，不影响结果）
    model.eval()
    return model


def main():
    config = Config()
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    transformers.set_seed(config.seed)

    accelerator = Accelerator()
    accelerator.print('Adaptor Dir : ', config.lora_weights_path)
    accelerator.print('Candidates  : ', CAND_SOURCE, CAND_PKL)

    for path in (config.lora_weights_path, CONCEPT_CSV, K_EMB_PATH, U_EMB_PATH,
                 TEST_DATA, CAND_PKL):
        if not os.path.exists(path):
            raise FileNotFoundError(f'路径不存在（脚本顶部常量改这里）: {path}')

    instruction_prompt, history_prompt = get_prompt()
    concept_df = pd.read_csv(CONCEPT_CSV, header=None, names=['index', 'id'])
    accelerator.print('All Concepts Num: ', len(concept_df))
    id2title = dict(zip(concept_df['index'], concept_df['id']))
    id2title_dict = {k: v.replace('K_', '') for k, v in id2title.items()}

    k_embs = np.load(K_EMB_PATH).astype(np.float32, copy=False)
    u_embs = np.load(U_EMB_PATH).astype(np.float32, copy=False)
    rows = pkl.load(open(TEST_DATA, 'rb'))
    accelerator.print(f'e_i={k_embs.shape}  x^u={u_embs.shape}  rows={len(rows)}')

    test_data = generate_list_from_pkl(
        items=build_items(CAND_SOURCE, CAND_PKL, rows),
        k_embs=k_embs,
        u_embs=u_embs,
        id2title_dict=id2title_dict,
        system_prompt=instruction_prompt,
        input_prompt=history_prompt,
    )
    if MAX_SAMPLES:
        test_data = test_data[:MAX_SAMPLES]
    accelerator.print(f'test samples: {len(test_data)}')

    result_json_data = os.path.join(config.lora_weights_path, f'predict_{CAND_SOURCE}.json')
    if os.path.exists(result_json_data):
        accelerator.print(f'The {result_json_data} has existed.（删掉它才能重跑）')
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

    def evaluate(instructions, inputs, candidate_lists, user_emb_list, his_emb_list,
                 candidate_emb_list, num_beams=config.num_beams,
                 max_new_tokens=MAX_NEW_TOKENS):
        prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        # 官方 A-LLMRec 不截断（左 padding + 截断会把 <user_des> 和历史切掉），这里同样不截断
        inputs = tokenizer(prompt, return_tensors='pt', padding=True,
                           truncation=False).to(accelerator.device)

        max_video_len = max([video.shape[0] for video in his_emb_list])
        for i in range(len(his_emb_list)):
            ev = his_emb_list[i]
            if ev.shape[0] < max_video_len:
                padding = np.zeros((max_video_len - ev.shape[0], ev.shape[1]))
                ev = np.concatenate([padding, ev], axis=0)
            his_emb_list[i] = ev

        inputs['user_emb'] = torch.from_numpy(np.stack(user_emb_list, axis=0)).to(accelerator.device)
        inputs['his_embs'] = torch.from_numpy(
            np.stack(his_emb_list, axis=0)).to(accelerator.device)
        inputs['candidate_embs'] = torch.from_numpy(
            np.stack(candidate_emb_list, axis=0)).to(accelerator.device)

        # 占位符个数必须和向量行数一一对应，否则 get_embs 的整位替换会串行（静默错位）
        n_user = int((inputs['input_ids'] == model.user_token_id).sum())
        n_his = int((inputs['input_ids'] == model.his_token_id).sum())
        n_con = int((inputs['input_ids'] == model.concept_token_id).sum())
        expect_his = int(np.sum([np.any(h != 0, axis=-1).sum() for h in his_emb_list]))
        expect_con = sum(len(c) for c in candidate_lists)
        assert n_user == len(instructions), f'<user_des> {n_user} != {len(instructions)}'
        assert n_his == expect_his, f'<his_des> {n_his} != his_embs 非零行 {expect_his}'
        assert n_con == expect_con, f'<concept_des> {n_con} != 候选数 {expect_con}'

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
        'user_emb_list': [_['user_emb'] for _ in test_data],
        'his_emb_list': [_['his_embs'] for _ in test_data],
        'candidate_emb_list': [_['candidate_embs'] for _ in test_data],
    }

    with accelerator.split_between_processes(input_dict) as input_temp:
        outputs, sequences_scores = [], []

        for batch1 in tqdm(
                zip(batch(input_temp['instructions']), batch(input_temp['inputs']),
                    batch(input_temp['candidate_lists']), batch(input_temp['user_emb_list']),
                    batch(input_temp['his_emb_list']), batch(input_temp['candidate_emb_list'])),
                total=(len(input_temp['instructions']) + config.batch_size - 1) // config.batch_size,
        ):
            (instructions, inputs, candidate_lists, user_emb_list, his_emb_list,
             candidate_emb_list) = batch1
            output, sequences_score = evaluate(instructions, inputs, candidate_lists,
                                               user_emb_list, his_emb_list, candidate_emb_list)
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
            test_data[i]['user_emb'] = None
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
                       'num_his': NUM_HIS, 'metrics': metrics}, f, ensure_ascii=False, indent=2)
        print(f"[{CAND_SOURCE}] " + '  '.join(
            f'{k}={v:.5f}' if isinstance(v, float) else f'{k}={v}' for k, v in metrics.items()))
        print(f'saved: {result_json_data}')


if __name__ == '__main__':
    fire.Fire(main)
