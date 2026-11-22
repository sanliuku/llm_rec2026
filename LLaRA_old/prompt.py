import pickle as pkl

def get_prompt():
    instruction_prompt = "Given a list of concepts the user has learned before, please recommend a new concept from the candidates that the user likes to the user."
    history_prompt = "The user has learned the following concepts before: {historical_concepts}\nThe candidates you need to select: {candidates}\nPlease return your choice."
    return instruction_prompt, history_prompt

def generate_prompt(instruction, warm_input, input):
    prompt_input_warm = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Input:
{warm_input}

### Response:
"""
    prompt_input = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    return prompt_input_warm, prompt_input

def generate_list_from_pkl(data_path, vk_dict, Emb_k, id2title_dict, system_prompt, input_prompt, num_his=20):
    test_data = pkl.load(open(data_path, 'rb'))
    json_data = []

    for index, data in enumerate(test_data):
        his_concepts = []
        for v in data[1]:
            his_concepts.extend(vk_dict[v])
        his_concepts = list(set(his_concepts))

        his_videos = data[1][-10:]
        current_concepts = []
        for v in his_videos:
            current_concepts.extend(vk_dict[v])
        current_concepts = current_concepts[-num_his:]
        his_embs = Emb_k[current_concepts].reshape(-1, Emb_k.shape[1])

        label = data[2]

        seq_t = ','.join([f'{i+1}.{id2title_dict[current_concepts[i]]} <PH>' for i in range(len(current_concepts))])
        seq_t2 = ','.join([f'{i+1}.{id2title_dict[current_concepts[i]]} <his_des>' for i in range(len(current_concepts))])

        input_prompt1 = input_prompt.format(historical_concepts=seq_t, candidates='<candidates>')
        input_prompt2 = input_prompt.format(historical_concepts=seq_t2, candidates='<candidates>')
        label_str = id2title_dict[label]

        json_entry = {'instruction': system_prompt, 'warm_input': input_prompt1, 'input': input_prompt2, 'output': label_str, 'uid': data[0], 'pid': label, 'video_label': data[3],'his_embs': his_embs, 'his_concepts': his_concepts}
        json_data.append(json_entry)
    return json_data