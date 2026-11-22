import torch
import numpy as np
import pickle as pkl
import pandas as pd
from tqdm import tqdm

column_names = ['id', 'name']
users = pd.read_csv('need_data/entities/users.csv', header=None)
users.columns = column_names
num_users = len(users)
videos = pd.read_csv('need_data/entities/videos.csv', header=None)
videos.columns = column_names
num_videos = len(videos)
courses = pd.read_csv('need_data/entities/courses.csv', header=None)
courses.columns = column_names
num_courses = len(courses)
concepts = pd.read_csv('need_data/entities/concepts.csv', header=None)
concepts.columns = column_names
num_concepts = len(concepts)

# adj
adj = torch.zeros(num_users, num_concepts)
with open('need_data/relations/adj.p', 'rb') as f:
    adjs = pkl.load(f)
for uk in tqdm(adjs):
    for k in adjs[uk]:
        adj[uk][k] = 1
print('adj:', adj.shape)
adj = adj.numpy()
np.save('need_data/matrix/adj.npy', adj)

# rate
rate = torch.zeros(num_users, num_concepts)
with open('need_data/relations/rate.p', 'rb') as f:
    rates = pkl.load(f)
for uk in tqdm(rates):
    for k in rates[uk]:
        rate[uk][k] = 1
print('rate:', rate.shape)
rate = rate.numpy()
np.save('need_data/matrix/rate.npy', rate)

ua_lastone = torch.zeros(num_users, num_concepts)
with open('need_data/relations/ua_lastone.p', 'rb') as f:
    ua_lastones = pkl.load(f)
for u in tqdm(ua_lastones):
    k = ua_lastones[u]
    ua_lastone[u][k] = 1
print('ua_lastone:', ua_lastone.shape)
ua_lastone = ua_lastone.numpy()
np.save('need_data/matrix/ua_lastone.npy', ua_lastone)

# adj = torch.from_numpy(adj)
# rate = adj - ua_lastone
# print(len(torch.nonzero((rate == 0).all(dim=1)).squeeze().tolist()))
# print(torch.where(rate < 0))
# rate = rate.numpy()
# np.save('need_data/matrix/rate.npy', rate)

# train_uc
train_uc = torch.zeros(num_users, num_courses)
with open('need_data/relations/train_uc.p', 'rb') as f:
    train_ucs = pkl.load(f)
for uc in tqdm(train_ucs):
    for c in train_ucs[uc]:
        train_uc[uc][c] = 1
print('train_uc:', train_uc.shape)
train_uc = train_uc.numpy()
np.save('need_data/matrix/train_uc.npy', train_uc)

# train_val_uc
train_val_uc = torch.zeros(num_users, num_courses)
with open('need_data/relations/train_val_uc.p', 'rb') as f:
    train_val_ucs = pkl.load(f)
for uc in tqdm(train_val_ucs):
    for c in train_val_ucs[uc]:
        train_val_uc[uc][c] = 1
print('train_val_uc:', train_val_uc.shape)
train_val_uc = train_val_uc.numpy()
np.save('need_data/matrix/train_val_uc.npy', train_val_uc)

# train_uv
train_uv = torch.zeros(num_users, num_videos)
with open('need_data/relations/train_uv.p', 'rb') as f:
    train_uvs = pkl.load(f)
for uv in tqdm(train_uvs):
    for v in train_uvs[uv]:
        train_uv[uv][v] = 1
print('train_uv:', train_uv.shape)
train_uv = train_uv.numpy()
np.save('need_data/matrix/train_uv.npy', train_uv)

# train_val_uv
train_val_uv = torch.zeros(num_users, num_videos)
with open('need_data/relations/train_val_uv.p', 'rb') as f:
    train_val_uvs = pkl.load(f)
for uv in tqdm(train_val_uvs):
    for v in train_val_uvs[uv]:
        train_val_uv[uv][v] = 1
print('train_val_uv:', train_val_uv.shape)
train_val_uv = train_val_uv.numpy()
np.save('need_data/matrix/train_val_uv.npy', train_val_uv)

# course-video
cv = torch.zeros(num_courses, num_videos)
with open('need_data/relations/cv.p', 'rb') as f:
    course_video = pkl.load(f)
for c in tqdm(course_video):
    for v in course_video[c]:
        cv[c][v] = 1
print('cv:', cv.shape)
cv = cv.numpy()
np.save('need_data/matrix/cv.npy', cv)

# course-concept
ck = torch.zeros(num_courses, num_concepts)
with open('need_data/relations/ck.p', 'rb') as f:
    course_concept = pkl.load(f)
for c in tqdm(course_concept):
    for k in course_concept[c]:
        ck[c][k] = 1
print('ck:', ck.shape)
ck = ck.numpy()
np.save('need_data/matrix/ck.npy', ck)

# video-concept
vk = torch.zeros(num_videos, num_concepts)
with open('need_data/relations/vk.p', 'rb') as f:
    video_concept = pkl.load(f)
for v in tqdm(video_concept):
    for k in video_concept[v]:
        vk[v][k] = 1
print('vk:', vk.shape)
vk = vk.numpy()
np.save('need_data/matrix/vk.npy', vk)

# user-action
ua = torch.zeros(num_users, num_concepts)
with open('need_data/relations/ua_dict.p', 'rb') as f:
    ua_dict = pkl.load(f)
for u in tqdm(ua_dict):
    for k in ua_dict[u]:
        ua[u][k] = 1
print('ua:', ua.shape)
ua = ua.numpy()
np.save('need_data/matrix/ua.npy', ua)

# ====================================================================================================
print('ok!!!!!!!!!!!!!')
