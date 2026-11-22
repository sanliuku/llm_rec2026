import asyncio
import json

import numpy as np

import pandas as pd
import pickle as pkl

from sentence_transformers import SentenceTransformer
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

embed_model = SentenceTransformer('/home/wbbmin/local_llm/bge-large-zh-v1.5')
for param in embed_model.parameters():
    param.requires_grad = False
embedding_size = 1024
concept_titles = []
for index in range(len(index2concept)):
    concept_titles.append(concept2title[index2concept[index]])

def generate_embs(texts):
    # tokens = embed_model.tokenize(texts)
    # features = embed_model({'input_ids': tokens['input_ids'].cuda(),
    #                         'attention_mask': tokens['attention_mask'].cuda()
    #                         })['sentence_embedding']
    features = embed_model.encode(texts)
    # features = features.cpu().detach().numpy()
    return features

total_features = []
batch = 64
for i in tqdm(range(0, len(concept_titles), batch)):
    texts = concept_titles[i:i + batch]
    features = generate_embs(texts)
    total_features.append(features)
total_features = np.concatenate(total_features, axis=0)
print(total_features.shape)

# save
np.save('need_data/embs/concept_embs_local.npy', total_features)
print('concept features saved.')