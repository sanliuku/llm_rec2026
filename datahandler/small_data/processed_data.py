import collections
import json
import pickle as pkl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import random

np.random.seed(0)
random.seed(0)

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

# load user_video_action
with open("../additional_information/user_video_act.json", "r") as f:
    lines = f.readlines()

# load course video
cv_df = pd.read_csv("../relations/course-video.json",
                    header=None,
                    delimiter="\t",
                    names=["course", "video"])
cv_dict = cv_df.groupby("course")["video"].apply(list).to_dict()

def get_total_df():
    user_dfs = list()
    '''
    {"id": "U_8651790",
     "activity":
         [{"course_id": "C_course-v1:TsinghuaX+30240184+sp",
           "video_id": "V_0a9f7e9606d64d16b3fa49469532e501",
           "watching_count": 25,
           "video_duration": 212.0,
           "local_watching_time": 266,
           "video_progress_time": 332.7999854087829,
           "video_start_time": 0.0,
           "video_end_time": 200.9900054931641,
           "local_start_time": "2018-08-22 09:37:24",
           "local_end_time": "2018-08-22 09:45:41"}]}
    '''
    for l in tqdm(lines):
        json_data = json.loads(l)
        activities = json_data["activity"]
        course_ids = list()
        video_ids = list()
        local_start_times = list()
        concepts = list()

        for a in activities:
            if a["course_id"] in ck_dict:
                course_ids.append(a["course_id"])
            else:
                # course_ids.append("")
                course_ids.append(None)
            video_ids.append(a["video_id"])
            local_start_times.append(a["local_start_time"])
            # concepts.append(vk_dict[a["video_id"]]) if a["video_id"] in vk_dict else concepts.append([""])
            concepts.append(vk_dict[a["video_id"]]) if a["video_id"] in vk_dict else concepts.append(None)

        df = pd.DataFrame({
            "id": json_data["id"],
            "cid": course_ids,
            "vid": video_ids,
            "concepts": concepts,
            "local_start_time": local_start_times
        })

        df.sort_values(by=['local_start_time', 'id'], inplace=True)
        user_dfs.append(df)

    # convert to total df
    total_df = pd.concat(user_dfs, ignore_index=True)
    # 删除concepts为空的行
    total_df = total_df.dropna()
    print(total_df.shape)
    return total_df

# total_df = get_total_df()
# pkl.dump(total_df, open("../total_df2.pkl", "wb"))
total_df = pkl.load(open("../total_df2.pkl", "rb"))
print(total_df["local_start_time"].max())
print(total_df["local_start_time"].min())

# filter the users before train_date_time
train_date_time = '2019-01-01 00:00:00'
bad_formatted_users = list(total_df[total_df["local_start_time"] < train_date_time]['id'].unique())
user_in_total_df = list(total_df['id'].unique())
selected_users = list(set(user_in_total_df)-set(bad_formatted_users))
len(selected_users)
filtered_df = total_df[total_df['id'].isin(selected_users)]
print(filtered_df.shape)
selected_df = filtered_df.copy()
# remove the same consecutive rows
retain_rows = []
pre_row = ''
for index, row in selected_df.iterrows():
    user_id = row['id']
    video_id = row['vid']
    identity = user_id + '|' + video_id
    if identity == pre_row:
        continue
    else:
        pre_row = identity
        retain_rows.append(row)
filtered_df = pd.DataFrame(retain_rows)
print(filtered_df.shape)
# ======================================================================================================================
# make sure the user can be found in both training and test period
test_date_start = '2019-05-01 00:00:00'
test_date_end = '2019-06-01 00:00:00'

user_in_trainingperiod = list(filtered_df[(filtered_df["local_start_time"] >= train_date_time) & (
        filtered_df["local_start_time"] < test_date_start)]['id'].unique())
user_in_testperiod = list(filtered_df[(filtered_df["local_start_time"] >= test_date_start) & (filtered_df["local_start_time"] < test_date_end)]['id'].unique())
selected_users = [u for u in user_in_testperiod if u in user_in_trainingperiod]
print(len(selected_users))
filtered_df = filtered_df[filtered_df["id"].isin(selected_users)]
################################################
###############################################
# 以前都是直接从某个时间拉到最后，没有考虑filter_df的时间问题 没有啥事
# 现在设置了test_time_end. 切记过滤掉时间之后的用户行为
filtered_df = filtered_df[(filtered_df["local_start_time"] >= train_date_time) & (filtered_df["local_start_time"] < test_date_end)]
###############################################
##############################################

# ======================================================================================================================
# design the main rule to filter the df
# check whether there is at least one concept in the testing period that is not in the training period
need_to_filter = list()
for u in selected_users:
    udf = filtered_df[filtered_df["id"] == u].copy()
    course_udf = udf["cid"].values.tolist()

    train_udf = udf[
        (udf["local_start_time"] >= train_date_time) & (udf["local_start_time"] < test_date_start)]
    train_udf = train_udf.sort_values('local_start_time')
    test_udf = udf[
        (udf["local_start_time"] >= test_date_start) & (udf["local_start_time"] < test_date_end)]
    test_udf = test_udf.sort_values('local_start_time')
    train_concepts = set([x for sublist in train_udf["concepts"].values for x in sublist if x != ""])
    test_concepts = set([x for sublist in test_udf["concepts"].values for x in sublist if x != ""])
    if len(test_concepts - train_concepts) == 0 or '' in course_udf:
        need_to_filter.append(u)
        continue

    train_courses = train_udf["cid"].values.tolist()
    test_courses = test_udf["cid"].values.tolist()
    train_videos = train_udf["vid"].values.tolist()
    val_video = train_videos[-1]

    if val_video not in vk_dict:
        need_to_filter.append(u)
        continue
    val_concepts = set(vk_dict[val_video])

    train_videos.pop(-1)
    train_new_concepts = set([x for v in train_videos if v in vk_dict for x in vk_dict[v]])

    test_videos = test_udf["vid"].values.tolist()
    test_video = test_videos[0]
    if test_video not in vk_dict:
        need_to_filter.append(u)
        continue
    test_new_concepts = set(vk_dict[test_video])
    # make sure inter>=6 val_video != test_video !in train_videos (concepts)
    if (val_video in train_videos) or (test_video in train_videos) or (val_video == test_video) or (len(train_videos) < 6)\
            or (len(test_new_concepts - (train_new_concepts | val_concepts)) == 0) or (len(val_concepts - train_new_concepts) == 0):
        need_to_filter.append(u)

need_to_filter = list(set(need_to_filter))
filtered_users = [u for u in selected_users if u not in need_to_filter]
filtered_df = filtered_df[filtered_df["id"].isin(filtered_users)]
print(filtered_df.shape)
pkl.dump(filtered_df, open("filtered_df.pkl", "wb"))
filtered_df = pkl.load(open('filtered_df.pkl', 'rb'))
# =======================================================================================================================
print("---------------- Entity Statistics -----------------")
distinct_concepts = list(set([c for sublist in filtered_df["concepts"].values for c in sublist if len(c) > 0]))

print("filtered df", filtered_df.shape)
print("total users", len(filtered_df["id"].unique()))
print("total courses", len(filtered_df["cid"].unique()))
print("total videos", len(filtered_df["vid"].unique()))
print("distinct concepts", len(distinct_concepts))

# =======================================================================================================================
# 统计每个用户的行为序列
train_val_df = filtered_df[(filtered_df["local_start_time"] >= train_date_time) & (filtered_df["local_start_time"] < test_date_start)]
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

'''
filtered df (112235, 5)
total users 1536
total courses 557
total videos 17660
distinct concepts 18858
'''