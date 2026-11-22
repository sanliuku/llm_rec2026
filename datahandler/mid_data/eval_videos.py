import pickle as pkl
import pandas as pd

path = 'need_data/'
# load user, video, concept data
columns = ['index', 'id']
user_df = pd.read_csv(path + 'entities/users.csv', header=None)
user_df.columns = columns
user2index = dict(zip(user_df['id'], user_df['index']))
video_df = pd.read_csv(path + 'entities/videos.csv', header=None)
video_df.columns = columns
video2index = dict(zip(video_df['id'], video_df['index']))
concept_df = pd.read_csv(path + 'entities/concepts.csv', header=None)
concept_df.columns = columns
concept2index = dict(zip(concept_df['id'], concept_df['index']))

train_date_time = '2018-01-01 00:00:00'
test_date_start = '2018-07-01 00:00:00'
test_date_end = '2018-08-01 00:00:00'

filtered_df = pkl.load(open('filtered_second_df2.pkl', 'rb'))
filtered_df['id'] = filtered_df['id'].apply(lambda x: user2index[x])
filtered_df['vid'] = filtered_df['vid'].apply(lambda x: video2index[x])
filtered_df = filtered_df.sort_values(by=['id', 'local_start_time'])
train_val_df = filtered_df[(filtered_df["local_start_time"] < test_date_start) & (filtered_df["local_start_time"] >= train_date_time)]
test_df = filtered_df[(filtered_df['local_start_time'] < test_date_end) & (filtered_df["local_start_time"] >= test_date_start)]
print('loading data start....')

train_videos = []
val_videos = []
test_videos = []
for uid in filtered_df['id'].unique():
    udf = filtered_df[filtered_df['id'] == uid]
    videos = [vid + 1 for vid in udf['vid'].values.tolist()]
    assert videos[-1] not in videos[:-2]
    assert videos[-2] not in videos[:-2]
    train_videos.extend(videos[:-2])
    val_videos.append(videos[-2])
    test_videos.append(videos[-1])
train_videos = list(set(train_videos))
val_videos = list(set(val_videos))
test_videos = list(set(test_videos))

print(len(set(val_videos) - set(train_videos)))
print(len(set(test_videos) - set(train_videos)))