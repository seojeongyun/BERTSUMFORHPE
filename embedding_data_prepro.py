import torch
from pprint import pprint as pp

dataset = torch.load('./bert_data/origin/all_arcface_768.tar')
# tgt = exercise name, src = 1 video = 16frames = 20 joints embeddings(20,512)
view_point = 0 # [0, 1, 2, ,3, 4]

for exercise_idx in dataset.keys():
    exer_video = dataset[exercise_idx]['Video']
    for video_idx in range(len(exer_video.keys())):
        try:
            data = {}
            frame_v = []
            video = exer_video[video_idx][view_point]
            for frame in video.keys():
                v = []
                for j in range(len(video[frame][0])):
                    if j > 1:
                        if video[frame][0][j].shape[-1] == 768:
                            v.append(video[frame][0][j])
                assert len(v) == 20
                frame_v.append(v)
            data['src'] = frame_v
            data['tgt'] = exercise_idx
            torch.save(data, './bert_data/exercise{}_videoIdx{}_viewPoint{}.pt'.format(exercise_idx,video_idx,view_point))
        except KeyError as e:
            print(f"KeyError: exercise {exercise_idx}, video {video_idx}, view_point {view_point} ? {e}")
            continue
        except Exception as e:
            print(f"Unexpected error at exercise {exercise_idx}, video {video_idx}: {e}")
            continue