import collections
import json

file_lines = json.load(open(r'E:\project\AAA_Important_model\llm_rec2026\LLaRA_old\runs\mid_data\results\f04e2f186abc3d328cb6a868fb82461d\predict_sasrec_global.json', 'r', encoding='utf-8'))
predict_items = [['K_' + title for title in top_K_items["predict"]] for top_K_items in file_lines]
candidate_list = [['K_' + title for title in top_K_items["candidate_list"]] for top_K_items in file_lines]
# load id2concept
import pandas as pd

concept_df = pd.read_csv(r'E:\project\AAA_Important_model\llm_rec2026\data\MOOCCube\mid_data\need_data\entities\concepts.csv')
concept_df.columns = ['id', 'name']
concept2index = dict(zip(concept_df['name'], concept_df['id']))

# load adj ua
import numpy as np
import torch

adj = np.load(r'E:\project\AAA_Important_model\llm_rec2026\data\MOOCCube\mid_data\need_data\matrix\adj.npy')
ua = np.load(r'E:\project\AAA_Important_model\llm_rec2026\data\MOOCCube\mid_data\need_data\matrix\ua.npy')
test_ua = ua - adj
test_ua = torch.from_numpy(test_ua)
test_user_dict = collections.defaultdict(set)
for x in torch.nonzero(test_ua).tolist():
    test_user_dict[x[0]].add(x[1])

def precision_k(k):
    precisions = []
    topk_items = predict_items
    # topk_items = candidate_list
    for i in range(1000):
        true_items = test_user_dict[i]
        pred_items = set([concept2index[item] for item in topk_items[i][:k]])
        num_hit = len(pred_items & true_items)
        precisions.append(num_hit / k)
    return sum(precisions) / len(precisions)

def recall_k(k):
    recalls = []
    topk_items = predict_items
    # topk_items = candidate_list

    for i in range(1000):
        pred_items = set([concept2index[item] for item in topk_items[i][:k]])
        true_items = test_user_dict[i]
        num_hit = len(pred_items & true_items)
        if len(true_items) == 0:
            recalls.append(0)
        else:
            recalls.append(num_hit / len(true_items))
    recall = sum(recalls) / len(recalls)
    return recall

def f1score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

print('precision@5: ', precision_k(5))
print('recall@5: ', recall_k(5))
print('f1@5: ', f1score(precision_k(5), recall_k(5)))
print('precision@10: ', precision_k(10))
print('recall@10: ', recall_k(10))
print('f1@10: ', f1score(precision_k(10), recall_k(10)))
