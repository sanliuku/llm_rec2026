import asyncio
import json

import numpy as np
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama
import pandas as pd
import pickle as pkl

from langchain_openai.chat_models.base import BaseChatOpenAI
from tqdm import tqdm

concept_df = pd.read_csv('need_data/entities/concepts.csv', header=None)
concept_df.columns = ['index', 'id']
concepts = concept_df['id'].tolist()
index2concept = dict(zip(concept_df['index'], concept_df['id']))
concept2title = pkl.load(open('need_data/aug_concepts.pkl', 'rb'))

total_concept_path = '../entities/concept.json'
total_concept_lines = open(total_concept_path, 'r', encoding='utf-8').readlines()
total_concept_jsons = [json.loads(l) for l in total_concept_lines]
total_concepts = {}
for concept in total_concept_jsons:
    if 'explanation' in concept and concept['explanation'] != '':
        total_concepts[concept['id']] = concept['explanation']
    else:
        continue

need_concepts = list(set(concepts) - set(concept2title.keys()))
print(len(set(concepts) - set(total_concepts.keys())))

# feature  augmentation
async def aschema_chat(llm, prompt_template, handel_result, default_result):
    work_memory = [HumanMessage(prompt_template)]
    while True:
        try:
            result = await llm.ainvoke(work_memory)
            break
        except Exception as e:
            print(e)
            await asyncio.sleep(60)
    result = result.content
    count_threshold = 3
    init_index = 0
    while True:
        try:
            result = handel_result(result)
            break
        except Exception as e:
            work_memory.append(AIMessage(result))
            work_memory.append(HumanMessage(f"输出格式不对，请按照要求输出。不需要额外的解释。错误提示：{e}"))
            while True:
                try:
                    result = await llm.ainvoke(work_memory)
                    break
                except Exception as e:
                    print(e)
                    await asyncio.sleep(60)
            result = result.content
            init_index += 1
            if init_index > count_threshold:
                result = default_result
                break
            continue
    return result


seed = 0
top_k = 40
top_p= 0.7
temperature = 0.2
repetition_penalty= 1.2
num_predict = 512
key = ''

llm = BaseChatOpenAI(
    model='deepseek-ai/DeepSeek-V3',
    openai_api_key=key,
    openai_api_base='https://api.siliconflow.cn/v1/',
    seed=seed,
    max_tokens=num_predict,
    top_p=top_p,
    temperature=temperature,
    max_retries=3,
)

template = '''你是一个概念定义专家，现在我给你一个概念的人工定义,我需要你对原有的定义进行数据增强。
概念名称：{concept_name}
人工定义：{human_definition}
注意：
1、 输出格式为：
{{\"definition\": \"增强后的定义\"}}
2、注意概念的名称可能出现在不同语境。
3、请结合你自身的知识增强人工定义。
4、增强后的定义应该以人工定义为基础，将已有的信息包装为概念属性，例如学科。另外，增强后的定义应该更加具体，更加清晰。
5、增强后的定义一定不能包含出处，例如书名、作者等信息。
6、增强后的定义不需要举例说明。
7、增强后的定义长度需在150字以内。'''
prompt = PromptTemplate(
    template=template,
    input_variables=['concept_name', 'human_definition'],
)

def handle_result(result):
    result_dict = json.loads(result)
    return result_dict['definition']

async def apredict(prompt, human_definition):
    result = await aschema_chat(llm, prompt, handle_result, human_definition)
    return result


async def apredict_batch():
    batch_size = 10
    for i in tqdm(range(0, len(need_concepts), batch_size)):
        concept_batch = need_concepts[i:i + batch_size]
        human_definition_batch = [total_concepts[concept] for concept in concept_batch]
        prompts = [prompt.format(**{
            'concept_name': concept,
            'human_definition': human_definition
        }) for concept, human_definition in zip(concept_batch, human_definition_batch)]
        predict_methods = [apredict(p, human_definition) for p, human_definition in zip(prompts, human_definition_batch)]
        results = await asyncio.gather(*predict_methods)
        concept2title.update(dict(zip(concept_batch, results)))
        pkl.dump(concept2title, open('need_data/aug_concepts.pkl', 'wb'))

# asyncio.run(apredict_batch())
# print('finish')
embed_model = OllamaEmbeddings(model="bge-m3:latest", base_url='http://localhost:11434')
embedding_size = 1024
concept_titles = []
for index in range(len(index2concept)):
    concept_titles.append(concept2title[index2concept[index]])
features = embed_model.embed_documents(concept_titles)
features = np.array(features).reshape(len(features), -1)
print(features.shape)

# save
np.save('need_data/embs/concept_embs.npy', features)
print('concept features saved.')