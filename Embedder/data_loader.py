from pprint import pprint
import pickle
import tqdm
import torch

from torch.utils.data import Dataset
from Embedder.Embedder_config import config
from tqdm import tqdm

class Video_Loader(Dataset):
    def __init__(self, config, mode):
        self.config = config
        self.mode = mode
        self.data_path = self.get_data_path()
        self.videos = self.get_data()

        # self.vocab = self.get_vocab()
        pprint('NUMBER OF VIDEOS:' + str(len(self.videos)))

    def get_data_path(self):
        if self.mode == 'train' or self.mode == 'train-valid':
            return self.config.T_DATA_PATH
        else:
            return self.config.V_DATA_PATH

    def get_data(self):
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        #
        pprint('VIDEO FILE SUCCESSFULLY LOADED USING PICKLE')
        return data

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video, workout_class = self.videos[idx]
        return video, workout_class

    def collate_fn(self, batch):
        videos, exercise_name = zip(*batch)
        exercise_name = torch.tensor(exercise_name) - 22
        return videos, exercise_name
        # return videos, exercise_name

if __name__ == '__main__':
    loader = Video_Loader(config)
    print('done')