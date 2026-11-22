import pickle
import random

import torch
import numpy as np

# load train_uk, test_uk, ua
from tqdm import tqdm

np.random.seed(0)
random.seed(0)

adj = np.load('need_data/matrix/adj.npy')
adj = torch.tensor(adj)
ua = np.load('need_data/matrix/ua.npy')
ua = torch.tensor(ua)
print('adj.shape:', adj.shape)
print('ua.shape:', ua.shape)
test_uk = ua - adj
print(torch.where(test_uk < 0))
test_uk = torch.where(test_uk > 0.5, 1, 0)
print('test_uk.shape:', test_uk.shape)

print(len(adj.nonzero().tolist()))
print(len(test_uk.nonzero().tolist()))

# get user-action lastone
ua_lastone = np.load('need_data/matrix/ua_lastone.npy')
ua_lastone = torch.tensor(ua_lastone)
print('ua_lastone.shape:', ua_lastone.shape)

# get user-item is none
def get_negativeIndex(h, g_h, k):
    # 求u-i为1的下标和为0的下标
    pos = h.nonzero().squeeze(1).tolist()
    neg = torch.where(g_h == 0)[0].tolist()

    # 对每个正样本匹配k个负样本
    negative_u = []
    for i in range(len(pos)):
        neg_i = []
        kneg = random.sample(neg, k=k)
        neg_i.extend(kneg)
        neg_i.append(pos[i])
        negative_u.append(neg_i)
    return negative_u

# get test_negative
'''
对test_uk中每个用户的一个用户记录+全局中他未交互的99个item，作为一个样本
negativeIndexs: 来自于全局
'''
def get_test_negative(test_uk, ua):
    num_u = test_uk.size(0)
    test_negative = {}
    for u in tqdm(range(num_u)):
        negative_u = get_negativeIndex(test_uk[u], ua[u], 19)
        test_negative[u] = negative_u
    print('test_negative[0]:', len(test_negative[0]), len(test_negative[0][0]))
    return test_negative


negative = get_test_negative(ua_lastone, adj)
with open('need_data/negatives/negative19.p', 'wb') as f:
    pickle.dump(negative, f)
test_negative = get_test_negative(test_uk, ua)
with open('need_data/negatives/test_negative19.p', 'wb') as f:
    pickle.dump(test_negative, f)

print('ok')