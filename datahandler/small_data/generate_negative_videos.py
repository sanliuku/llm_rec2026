import collections
import pickle as pkl
import random

import pandas as pd
import torch
import numpy as np

from tqdm import tqdm

np.random.seed(0)
random.seed(0)

adj = np.load('need_data/matrix/train_val_uv.npy')
adj = torch.tensor(adj)

train_date_time = '2019-01-01 00:00:00'
test_date_start = '2019-05-01 00:00:00'
test_date_end = '2019-06-01 00:00:00'
columns = ['index', 'id']
path = 'need_data/'
user_df = pd.read_csv(path + 'entities/users.csv', header=None)
user_df.columns = columns
user2index = dict(zip(user_df['id'], user_df['index']))
video_df = pd.read_csv(path + 'entities/videos.csv', header=None)
video_df.columns = columns
video2index = dict(zip(video_df['id'], video_df['index']))
concept_df = pd.read_csv(path + 'entities/concepts.csv', header=None)
concept_df.columns = columns
concept2index = dict(zip(concept_df['id'], concept_df['index']))

filtered_df = pkl.load(open('filtered_second_df2.pkl', 'rb'))
filtered_df['id'] = filtered_df['id'].apply(lambda x: user2index[x])
filtered_df['vid'] = filtered_df['vid'].apply(lambda x: video2index[x])
filtered_df = filtered_df.sort_values(by=['id', 'local_start_time'])
train_val_df = filtered_df[(filtered_df["local_start_time"] < test_date_start) & (filtered_df["local_start_time"] >= train_date_time)]
test_df = filtered_df[(filtered_df['local_start_time']<test_date_end) & (filtered_df["local_start_time"] >= test_date_start)]
print('loading data start....')

ua = collections.defaultdict(set)
for row in filtered_df.itertuples():
    user = row.id
    video = row.vid
    ua[user].add(video)
ua = {k: list(v) for k, v in ua.items()}

ua_matrix = np.zeros((adj.shape))
for user, videos in ua.items():
    for video in videos:
        ua_matrix[user][video] = 1
ua = torch.from_numpy(ua_matrix)

print('adj.shape:', adj.shape)
print('ua.shape:', ua.shape)
test_uk = ua - adj
print(torch.where(test_uk < 0))
test_uk = torch.where(test_uk > 0.5, 1, 0)
print('test_uk.shape:', test_uk.shape)

print(len(adj.nonzero().tolist()))
print(len(test_uk.nonzero().tolist()))

# get user-action lastone
rate = torch.from_numpy(np.load('need_data/matrix/train_uv.npy'))
ua_lastone = adj - rate
ua_lastone = torch.tensor(ua_lastone)
print('ua_lastone.shape:', ua_lastone.shape)
print(len(ua_lastone.nonzero().tolist()))

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
        negative_u = get_negativeIndex(test_uk[u], ua[u], 99)
        test_negative[u] = negative_u
    print('test_negative[0]:', len(test_negative[0]), len(test_negative[0][0]))
    return test_negative


negative = get_test_negative(ua_lastone, adj)
with open('need_data/negatives/negvideo99.p', 'wb') as f:
    pkl.dump(negative, f)
test_negative = get_test_negative(test_uk, ua)
with open('need_data/negatives/test_negvideo99.p', 'wb') as f:
    pkl.dump(test_negative, f)

print('ok')