import ast
import json
import os
import random
from typing import List

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
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import pickle as pkl

from pt_reasoner import PTReasoner

# test_negative = pkl.load(open('../data/MOOCCube/small_data/need_data/negatives/test_negative.p', 'rb'))
test_negative = pkl.load(open('../prepare_data/mooc_sasrec_augllm/test_new_negative.p', 'rb'))  #
print(len(test_negative[0][0]))
un_candi = {}
for i in test_negative:
    negs = test_negative[i]
    for neg in negs:
        key = str(i) + '|' + str(neg[-1])
        value = neg[:-1]
        # value = neg
        un_candi[key] = value

class Config(BaseModel):
    lora_weights_path: str = '/home/root/llm_rec/ALLMRec_new/runs/small_data/results/5b6e3cdbf9713eb51fb090ca7b099787'   #
    dataset: str = 'MOOC'
    test_sample: int = 5000
    embed_size: int = 256
    emb_dim: int = embed_size
    bert_dim: int = 1024
    token_size: int = 1536 # 1536
    num_virtual_tokens: int = 0
    cutoff_len: int = 1024
    hidden_size: int = 128
    
    batch_size: int = 16
    base_model: str = "../../local_llm/Qwen2.5-1.5B"
    num_beams: int = 10
    constraint_BS: int = 1
    seed: int = 12
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = [
    "q_proj",
    "v_proj",
    ]
    device: str = 'cuda:0'

def get_prompt(dataset_name):
    instruction_prompt, history_prompt = None, None
    if "MOOC" in dataset_name:
        instruction_prompt = "The description of user is: <user_des>. Given a list of concepts the user has learned before, please recommend a new concept from the candidates that the user likes to the user."
        history_prompt = "The user has learned the following concepts before: {historical_concepts}\nThe candidates you need to select: {candidates}\nPlease return your choice."

    return instruction_prompt, history_prompt

def generate_list_from_pkl(train_data_path, vk_dict, Emb_u, Emb_v, Emb_k, id2title_dict, instruction_str, input_prefix_str, num_items, num_his=20):
    train_data = pkl.load(open(train_data_path, 'rb'))
    json_data = []

    for index, data in enumerate(train_data):
        user_emb = Emb_u[index].reshape(1, -1)
        his_videos = data[1][-10:]
        current_concepts = []
        for v in his_videos:
            current_concepts.extend(vk_dict[v])
        current_concepts = current_concepts[-num_his:]

        his_embs = Emb_k[current_concepts].reshape(-1, Emb_k.shape[1])
        label = data[2]

        candidates = un_candi[str(data[0]) + '|' + str(label)]
        candidate_embs = Emb_k[candidates].reshape(-1, Emb_k.shape[1])

        candidate_list = [id2title_dict[candidate] for candidate in candidates]
        candidates_text = ', '.join([f'({id2title_dict[candidate]}, <concept_des>)' for candidate in candidates])
        input_str = input_prefix_str.format(
            **{'historical_concepts': ','.join([f'{i+1}.({id2title_dict[current_concepts[i]]}, <his_des>)' for i in range(len(current_concepts))]),
               'candidates': '<candidates>'}) \
            if 'candidates' in input_prefix_str else input_prefix_str.format(
            **{'historical_concepts': ','.join([f'{i+1}.({id2title_dict[current_concepts[i]]}, <his_des>)' for i in range(len(current_concepts))])})

        input_str = input_str.replace('<candidates>', candidates_text) if '<candidates>' in input_str else input_str
        label_str = f'{id2title_dict[label]}'

        json_entry = {'instruction': instruction_str, 'input': f'{input_str}', 'output': f'{label_str}', 'candidate_list': candidate_list, 'his_embs': his_embs, 'candidate_embs': candidate_embs, 'user_emb': user_emb}
        json_data.append(json_entry)
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

def main():

    config = Config()
    #################
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    transformers.set_seed(config.seed)

    accelerator = Accelerator()
    accelerator.print("Dataset: ", config.dataset)
    accelerator.print("LoRA Weights Path: ", config.lora_weights_path)
    data_path = os.path.join("data", config.dataset)
    test_data_path = os.path.join(data_path, f"test_{config.test_sample}.csv")
    accelerator.print("test_data_path: ", test_data_path)
    instruction_prompt, history_prompt = get_prompt(config.dataset)

    id2title_path = '../data/MOOCCube/small_data/need_data/entities/concepts.csv'
    columns = ['index', 'id']
    concept_df = pd.read_csv(id2title_path, header=None)
    concept_df.columns = columns
    num_items = len(concept_df)
    print('All Concepts Num: ', num_items)
    id2title = dict(zip(concept_df['index'], concept_df['id']))
    id2title_dict = {k: v.replace('K_', '') for k, v in id2title.items()}
    print(id2title_dict[1])
    v_embs = np.load('../prepare_data/mooc_sasrec_augllm/Emb_v.npy')   #
    k_embs = np.load('/home/weiliting/llm_rec/sasrec_new/runs/small_data/prepare_data/39ca25ef14821014de76d4dbb3ead1b8/Emb_k.npy')   #
    u_embs = np.load('/home/weiliting/llm_rec/sasrec_new/runs/small_data/prepare_data/39ca25ef14821014de76d4dbb3ead1b8/test_user_features.npy')  #
    vk_dict = pkl.load(open('../data/MOOCCube/small_data/need_data/relations/vk.p', 'rb'))

    test_data = generate_list_from_pkl(
        train_data_path='../prepare_data/eval_data_new.p',
        vk_dict=vk_dict,
        Emb_u=u_embs,
        Emb_v=v_embs,
        Emb_k=k_embs,
        id2title_dict=id2title_dict,
        instruction_str=instruction_prompt,
        input_prefix_str=history_prompt,
        num_items=num_items,
    )
    # print(test_data[0])
    # test_data = random.sample(test_data, 200)

    num_sample = config.test_sample
    result_json_data = f"predict_{config.dataset}_{num_sample}_CBS"
    result_json_data = os.path.join(config.lora_weights_path, result_json_data + ".json")
    if os.path.exists(result_json_data):
        accelerator.print(f"The {result_json_data} has existed.")
        return
    accelerator.wait_for_everyone()

    # load model
    model = PTReasoner(model_path=config.base_model, config=config, device=config.device, is_lora=True)
    model.load_model(config)
    model.eval()
    tokenizer = model.tokenizer
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # define prompt and construct train data
    assistant_sep = '### Response:\n'
    assistant_sep_token = tokenizer.encode(assistant_sep)
    end_sep = '<|endoftext|>'
    end_sep_token = tokenizer.encode(end_sep)

    # MarisaTrie
    # titles_list = list(id2title_dict.values())
    # tokens_list = [tokenizer.encode(assistant_sep + f'{title}') for title in titles_list]
    # print(tokens_list[0])
    #
    # trie = MarisaTrie(tokens_list)

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

    def evaluate(instructions, inputs, candidate_lists, his_emb_list, user_emb_list, candidate_emb_list, num_beams=config.num_beams, max_new_tokens=128):
        prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=config.cutoff_len).to(accelerator.device)

        max_video_len = max([video.shape[0] for video in his_emb_list])
        for i in range(len(his_emb_list)):
            ev = his_emb_list[i]
            if ev.shape[0] < max_video_len:
                padding = np.zeros((max_video_len - ev.shape[0], ev.shape[1]))
                ev = np.concatenate([padding, ev], axis=0)
            his_emb_list[i] = ev

        his_embs = np.stack(his_emb_list, axis=0)
        his_embs = torch.from_numpy(his_embs).to(accelerator.device)
        inputs['his_embs'] = his_embs
        user_emb = np.stack(user_emb_list, axis=0)
        user_emb = torch.from_numpy(user_emb).to(accelerator.device)
        inputs['user_emb'] = user_emb
        candidate_embs = np.stack(candidate_emb_list, axis=0)
        candidate_embs = torch.from_numpy(candidate_embs).to(accelerator.device)
        inputs['candidate_embs'] = candidate_embs

        input_embs = model.get_embs(inputs)
        # attention_mask = model.get_attention_mask(inputs)
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
                sequences_scores[i * num_beams : (i + 1) * num_beams] for i in range(len(sequences_scores) // num_beams)
            ]

            output_seq = generation_output.sequences
            output = tokenizer.batch_decode(output_seq, skip_special_tokens=False)
            output = [_.split(assistant_sep)[-1].split(end_sep)[0] for _ in output]
            real_outputs = [output[i * num_beams : (i + 1) * num_beams] for i in range(len(output) // num_beams)]
        return real_outputs, sequences_scores

    def batch(list, batch_size=config.batch_size):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i : batch_size * (i + 1)]

    instructions = [_["instruction"] for _ in test_data]
    inputs = [_["input"] for _ in test_data]
    candidate_lists = [_['candidate_list'] for _ in test_data]
    his_embs = [_['his_embs'] for _ in test_data]
    user_embs = [_['user_emb'] for _ in test_data]
    candidate_embs = [_['candidate_embs'] for _ in test_data]
    input_dict = {"instructions": instructions, "inputs": inputs, "candidate_lists": candidate_lists, "his_emb_list": his_embs, "user_emb_list": user_embs, "candidate_emb_list": candidate_embs}

    with accelerator.split_between_processes(input_dict) as input_temp:
        outputs = []
        sequences_scores = []

        for batch1 in tqdm(
            zip(batch(input_temp["instructions"]), batch(input_temp["inputs"]), batch(input_temp["candidate_lists"]), batch(input_temp["his_emb_list"]), batch(input_temp["user_emb_list"]), batch(input_temp["candidate_emb_list"])),
            total=(len(input_temp["instructions"]) + config.batch_size - 1) // config.batch_size,
        ):
            instructions, inputs, candidate_lists, his_emb_list, user_emb_list, candidate_emb_list = batch1
            output, sequences_score = evaluate(instructions, inputs, candidate_lists, his_emb_list, user_emb_list, candidate_emb_list)
            outputs.extend(output)
            sequences_scores.extend(sequences_score)

    outputs = gather_object(outputs)
    sequences_scores = gather_object(sequences_scores)
    assert len(outputs) == len(test_data)
    assert len(sequences_scores) == len(test_data)

    if accelerator.is_main_process:
        for i, _ in enumerate(test_data):
            test_data[i]["predict"] = outputs[i]
            test_data[i]["scores"] = sequences_scores[i]
            test_data[i]["his_embs"] = None
            test_data[i]["user_emb"] = None
            test_data[i]["candidate_embs"] = None

        with open(result_json_data, "w", encoding='utf-8') as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    fire.Fire(main)
