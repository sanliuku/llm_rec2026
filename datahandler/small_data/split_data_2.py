import collections

import pandas as pd
import numpy as np
import random
import pickle as pkl

from matplotlib import pyplot as plt

np.random.seed(0)
random.seed(0)
# =======================================================================================================================
# load video concepts
vk_df = pd.read_csv("../relations/video-concept.json",
                    header=None,
                    delimiter="\t",
                    names=["video", "concept"])
vk_dict = vk_df.groupby('video')['concept'].apply(list).to_dict()
print(vk_dict['V_9f3cc65bd68447b3b489be6ba8510a39'])

# load course concepts
ck_df = pd.read_csv("../relations/course-concept.json",
                    header=None,
                    delimiter="\t",
                    names=["course", "concept"])
ck_dict = ck_df.groupby('course')['concept'].apply(list).to_dict()
print(ck_dict['C_course-v1:KMUSTX+1803168+2019_T1'])

# load course video
cv_df = pd.read_csv("../relations/course-video.json",
                    header=None,
                    delimiter="\t",
                    names=["course", "video"])
cv_dict = cv_df.groupby("course")["video"].apply(list).to_dict()

filtered_df = pkl.load(open('filtered_df.pkl', 'rb'))
print("filtered df", filtered_df.shape)
filtered_df = filtered_df.reset_index(drop=True)
# =======================================================================================================================
print("---------------- Entity Statistics -----------------")
distinct_concepts = list(set([c for sublist in filtered_df["concepts"].values for c in sublist if len(c) > 0]))
print("total users", len(filtered_df["id"].unique()))
print("total courses", len(filtered_df["cid"].unique()))
print("total videos", len(filtered_df["vid"].unique()))
print("distinct concepts", len(distinct_concepts))
'''
(112235, 5)
---------------- Entity Statistics -----------------
filtered df (112235, 5)
total users 1536
total courses 557
total videos 17660
distinct concepts 18858
'''

# =======================================================================================================================
# random sample 100 users
users = filtered_df["id"].unique()
selected_users = random.sample(list(users), 1000)
filtered_second_df = filtered_df[filtered_df["id"].isin(selected_users)].copy()
print("filtered second df", filtered_second_df.shape)
filtered_df = filtered_second_df
print("filtered df", filtered_df.shape)
# =======================================================================================================================
# 统计每个用户的行为序列
user_behaviors = collections.defaultdict(set)
for row in filtered_df.itertuples():
    uid = row.id
    vid = row.vid
    user_behaviors[uid].add(vid)
user_behaviors = {k: list(v) for k, v in user_behaviors.items()}
video_uv_rate = {k: len(v) for k, v in user_behaviors.items()}
# 从小到大排序
xs = list(range(len(video_uv_rate)))
ys = sorted(video_uv_rate.values())
plt.plot(xs, ys)
plt.show()

# =======================================================================================================================
train_date_time = '2019-01-01 00:00:00'
test_date_start = '2019-05-01 00:00:00'
test_date_end = '2019-06-01 00:00:00'

filtered_df = filtered_df.sort_values(by='local_start_time')
filtered_df = filtered_df.reset_index(drop=True)
train_val_df = filtered_df[(filtered_df["local_start_time"] < test_date_start) & (filtered_df["local_start_time"] >= train_date_time)]
filtered_new_df = train_val_df.copy()
for uid in filtered_df["id"].unique():
    tv_udf = train_val_df[train_val_df["id"] == uid]
    total_udf = filtered_df[filtered_df["id"] == uid]
    test_udf = total_udf.iloc[len(tv_udf)]
    filtered_new_df = filtered_new_df._append(test_udf)
filtered_df = filtered_new_df.copy()
filtered_df = filtered_df.sort_values(by='local_start_time')
filtered_df = filtered_df.reset_index(drop=True)

pkl.dump(filtered_df, open('filtered_second_df2.pkl', 'wb'))
filtered_df = pkl.load(open('filtered_second_df2.pkl', 'rb'))

print("---------------- Entity Statistics -----------------")
distinct_concepts = list(set([c for sublist in filtered_df["concepts"].values for c in sublist if len(c) > 0]))
print("total users", len(filtered_df["id"].unique()))
print("total courses", len(filtered_df["cid"].unique()))
print("total videos", len(filtered_df["vid"].unique()))
print("distinct concepts", len(distinct_concepts))

'''
filtered df (75603, 5)
---------------- Entity Statistics -----------------
total users 1000
total courses 484
total videos 12264
distinct concepts 15827
'''
# =======================================================================================================================
ua = collections.defaultdict(set)
for row in filtered_df.itertuples():
    for k in row.concepts:
        if k!='':
            ua[row.id].add(k)
ua = {key: list(value) for key, value in ua.items()}
print('ua:', sum([len(x) for x in ua.values()]))
# =======================================================================================================================
users = filtered_df["id"].unique()
users = sorted(users)
user2index = dict(zip(users, range(len(users))))
concepts = sorted(distinct_concepts)
concept2index = dict(zip(concepts, range(len(concepts))))
courses = filtered_df["cid"].unique()
courses = sorted(courses)
course2index = dict(zip(courses, range(len(courses))))
videos = filtered_df["vid"].unique()
videos = sorted(videos)
video2index = dict(zip(videos, range(len(videos))))
users_df = pd.DataFrame([{"id": value, "name": key} for key, value in user2index.items()])
concepts_df = pd.DataFrame([{"id": value, "name": key} for key, value in concept2index.items()])
courses_df = pd.DataFrame([{"id": value, "name": key} for key, value in course2index.items()])
videos_df = pd.DataFrame([{"id": value, "name": key} for key, value in video2index.items()])
users_df.to_csv('need_data/entities/users.csv', index=False, header=False)
concepts_df.to_csv('need_data/entities/concepts.csv', index=False, header=False)
courses_df.to_csv('need_data/entities/courses.csv', index=False, header=False)
videos_df.to_csv('need_data/entities/videos.csv', index=False, header=False)

# =======================================================================================================================
train_date_time = '2019-01-01 00:00:00'
test_date_start = '2019-05-01 00:00:00'
test_date_end = '2019-06-01 00:00:00'

train_val_df = filtered_df[(filtered_df["local_start_time"] < test_date_start) & (filtered_df["local_start_time"] >= train_date_time)]
pkl.dump(train_val_df, open("train_val_df2.pkl", "wb"))
train_df = pd.DataFrame()
for user in train_val_df['id'].unique():
    udf = train_val_df[train_val_df['id'] == user]
    udf = udf.sort_values(by='local_start_time')
    train_df = pd.concat([train_df, udf.iloc[:-1]])
pkl.dump(train_df, open("train_df2.pkl", "wb"))
# =======================================================================================================================
# relationship
cv = {cid: list({v for v in cv_dict[cid] if v in videos}) for cid in courses if cid in cv_dict}
ck = {cid: list({k for k in ck_dict[cid] if k in distinct_concepts}) for cid in courses if cid in ck_dict}
vk = {vid: list({k for k in vk_dict[vid] if k in distinct_concepts}) for vid in videos if vid in vk_dict}
# =======================================================================================================================
filtered_df['id'] = filtered_df['id'].map(user2index)
filtered_df['cid'] = filtered_df['cid'].map(course2index)
filtered_df['vid'] = filtered_df['vid'].map(video2index)
filtered_df['concepts'] = filtered_df['concepts'].map(lambda x: [concept2index[c] for c in x if c!=''])

train_val_df['id'] = train_val_df['id'].map(user2index)
train_val_df['cid'] = train_val_df['cid'].map(course2index)
train_val_df['vid'] = train_val_df['vid'].map(video2index)
train_val_df['concepts'] = train_val_df['concepts'].map(lambda x: [concept2index[c] for c in x if c!=''])

train_df['id'] = train_df['id'].map(user2index)
train_df['cid'] = train_df['cid'].map(course2index)
train_df['vid'] = train_df['vid'].map(video2index)
train_df['concepts'] = train_df['concepts'].map(lambda x: [concept2index[c] for c in x if c!=''])
# =======================================================================================================================
# cv ck vk 重编码
ck = {course2index[key]: [concept2index[k] for k in value] for key, value in ck.items()}
vk = {video2index[key]: [concept2index[k] for k in value] for key, value in vk.items()}
cv = {course2index[key]: [video2index[k] for k in value] for key, value in cv.items()}

# 由于df生成时 k是通过v的dict来获取，所以可以直接在dataframe中获取两者关系
with open('need_data/relations/cv.p', 'wb') as f:
    pkl.dump(cv, f)
with open('need_data/relations/ck.p', 'wb') as f:
    pkl.dump(ck, f)
with open('need_data/relations/vk.p', 'wb') as f:
    pkl.dump(vk, f)
print('course-video-relationships:', sum([len(x) for x in cv.values()]))
print("course-concept-relationships:", sum([len(x) for x in ck.values()]))
print("video-concept-relationships:", sum([len(x) for x in vk.values()]))

# =======================================================================================================================
# get ua_dict, train_uk, test_uk, train_ucv, test_ucv
ua_dict = collections.defaultdict(set)
for row in filtered_df.itertuples():
    for k in row.concepts:
        if k!='':
            ua_dict[row.id].add(k)
ua_dict = {key: list(value) for key, value in ua_dict.items()}

# ua lastone
ua_lastone = collections.defaultdict(set)
for i in range(len(users)):
    udf_train = train_df[train_df['id'] == i].copy()
    train_concepts = set([x for sublist in udf_train["concepts"].values for x in sublist if x != ""])
    udf_train_val = train_val_df[train_val_df['id'] == i].copy()
    train_val_concepts = set([x for sublist in udf_train_val["concepts"].values for x in sublist if x != ""])
    ua_lastone[i] = train_val_concepts - train_concepts
ua_lastone = {key: list(value) for key, value in ua_lastone.items()}
print('ua_lastone:', sum([len(x) for x in ua_lastone.values()]))

rate = collections.defaultdict(set)
for row in train_df.itertuples():
    for k in row.concepts:
        if k!='':
            rate[row.id].add(k)
rate = {key: list(value) for key, value in rate.items()}
print('rate:', sum([len(x) for x in rate.values()]))

adj = collections.defaultdict(set)
for row in train_val_df.itertuples():
    for k in row.concepts:
        if k!='':
            adj[row.id].add(k)
adj = {key: list(value) for key, value in adj.items()}
print('adj:', sum([len(x) for x in adj.values()]))

uc = collections.defaultdict(set)
for row in filtered_df.itertuples():
    uc[row.id].add(row.cid)
uc = {key: list(value) for key, value in uc.items()}
print("user-course:", sum([len(x) for x in uc.values()]))

uv = collections.defaultdict(set)
for row in filtered_df.itertuples():
    uv[row.id].add(row.vid)
uv = {key: list(value) for key, value in uv.items()}
print("user-video:", sum([len(x) for x in uv.values()]))

train_uc = collections.defaultdict(set)
for row in train_df.itertuples():
    train_uc[row.id].add(row.cid)
train_uc = {key: list(value) for key, value in train_uc.items()}
print('train_uc:', sum([len(x) for x in train_uc.values()]))

train_val_uc = collections.defaultdict(set)
for row in train_val_df.itertuples():
    train_val_uc[row.id].add(row.cid)
train_val_uc = {key: list(value) for key, value in train_val_uc.items()}
print('train_val_uc:', sum([len(x) for x in train_val_uc.values()]))

train_uv = collections.defaultdict(set)
for row in train_df.itertuples():
    train_uv[row.id].add(row.vid)
train_uv = {key: list(value) for key, value in train_uv.items()}
print('train_uv:', sum([len(x) for x in train_uv.values()]))

val_uv = {}
for u in range(len(users)):
    udf = train_val_df[train_val_df['id'] == u]
    udf = udf.sort_values(by='local_start_time')
    val_video = udf['vid'].values.tolist()[-1]
    val_uv[u] = [val_video]
    val_concept = vk[val_video]
    last_concept = ua_lastone[u][0]
    assert last_concept in val_concept
print('train_uv:', sum([len(x) for x in train_uv.values()]))
print('val_uv:', sum([len(x) for x in val_uv.values()]))

train_val_uv = collections.defaultdict(set)
for row in train_val_df.itertuples():
    train_val_uv[row.id].add(row.vid)
train_val_uv = {key: list(value) for key, value in train_val_uv.items()}
print('train_val_uv:', sum([len(x) for x in train_val_uv.values()]))

test_uv = {}
test_df = filtered_df[(filtered_df['local_start_time'] >= test_date_start)&(filtered_df['local_start_time'] < test_date_end)]
for u in range(len(users)):
    udf = test_df[test_df['id'] == u]
    udf = udf.sort_values(by='local_start_time')
    test_uv[u] = udf['vid'].values.tolist()
print('test_uv:', sum([len(x) for x in test_uv.values()]))

with open('need_data/relations/ua_dict.p', 'wb') as f:
    pkl.dump(ua_dict, f)
with open('need_data/relations/ua_lastone.p', 'wb') as f:
    pkl.dump(ua_lastone, f)
with open('need_data/relations/rate.p', 'wb') as f:
    pkl.dump(rate, f)
with open('need_data/relations/adj.p', 'wb') as f:
    pkl.dump(adj, f)
with open('need_data/relations/train_uc.p', 'wb') as f:
    pkl.dump(train_uc, f)
with open('need_data/relations/train_uv.p', 'wb') as f:
    pkl.dump(train_uv, f)
with open('need_data/relations/val_uv.p', 'wb') as f:
    pkl.dump(val_uv, f)
with open('need_data/relations/test_uv.p', 'wb') as f:
    pkl.dump(test_uv, f)
with open('need_data/relations/train_val_uc.p', 'wb') as f:
    pkl.dump(train_val_uc, f)
with open('need_data/relations/train_val_uv.p', 'wb') as f:
    pkl.dump(train_val_uv, f)
print('user-concept-relationships:', sum([len(x) for x in ua_dict.values()]))

'''
ua: 243467
course-video-relationships: 16350
course-concept-relationships: 119422
video-concept-relationships: 119734
ua_lastone: 5313
rate: 231901
adj: 237214
user-course: 4002
user-video: 53039
train_uc: 3529
train_val_uc: 3606
train_uv: 51039
val_uv: 1000
train_val_uv: 52039
test_uv: 1000
user-concept-relationships: 243467
'''