import json
import pickle as pkl

import pandas as pd

concept2titles = pkl.load(open('need_data/aug_concepts.pkl', 'rb'))
print(list(concept2titles.values())[:5])

concept_df = pd.read_csv('need_data/entities/concepts.csv', header=None)
concept_df.columns = ['index', 'id']
concepts = concept_df['id'].tolist()


# with open('need_data/pt_concepts.json', 'w', encoding='utf-8') as f:
#     for concept in concepts:
#         kid = concept.split('K_')[1]
#         text = concept2titles[concept]
#         kdict = {'text': f'{kid}的定义是：{text}'}
#         f.write(json.dumps(kdict, ensure_ascii=False) + '\n')
# print('concept features saved.')
