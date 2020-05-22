from torchvision.datasets import VisionDataset
from PIL import Image
from math import ceil
import numpy as np
import random
import os
import sys
import torch

IMAGE = 0
LABEL = 1
TEST_USER = 'S2'


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class GTEA61(VisionDataset):
    def __init__(self, root, split='train', seq_len=16, transform=None, target_transform=None):
        super(GTEA61, self).__init__(root, transform=transform, target_transform=target_transform)
        self.datadir = root
        self.split = split
        self.seq_len = seq_len
        self.label_map = {}
        self.videos = []
        self.labels = []
        self.n_frames = []

        users = os.listdir(self.datadir)
        if len(users) != 4:
            raise FileNotFoundError("you specified the wrong directory")
        if TEST_USER not in users:
            raise FileNotFoundError("S2 folder not found")
        if self.split == 'test':
            folders = [users[users.index(TEST_USER)]]
        else:
            users.remove(TEST_USER)
            folders = users

        classes = os.listdir(os.path.join(self.datadir, folders[0]))
        self.label_map = {act: i for i, act in enumerate(classes)}
        for user in folders:
            user_dir = os.path.join(self.datadir, user)
            for action in os.listdir(user_dir):
                action_dir = os.path.join(user_dir, action)
                for element in os.listdir(action_dir):
                    frames = os.path.join(action_dir, element, "rgb")
                    self.videos.append(frames)
                    self.labels.append(self.label_map[action])
                    self.n_frames.append(len(os.listdir(frames)))

    def __getitem__(self, index):
        vid = self.videos[index]
        label = self.labels[index]
        length = self.n_frames[index]

        frames = np.array(sorted(os.listdir(vid)))
        select_indices = np.linspace(0, length, self.seq_len, endpoint=False, dtype=int)
        select_frames = frames[select_indices]
        select_files = [os.path.join(vid, frame) for frame in select_frames]
        sequence = [pil_loader(file) for file in select_files]
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            sequence = [self.transform(image) for image in sequence]
        sequence = torch.stack(sequence, 0)

        return sequence, label

    def __len__(self):
        return len(self.videos)

    def split_indices(self, size):
        taken = []
        left = []
        for_each_class = {key: [0, ceil(item[1] * size)] for key, item in self.dataset_by_key.items()}
        for i, item in enumerate(self.data_list):
            if for_each_class[item[LABEL]][0] < for_each_class[item[LABEL]][1]:
                taken.append(i)
                for_each_class[item[LABEL]][0] += 1
            else:
                left.append(i)

        return taken, left

# d = Caltech('Caltech101/101_ObjectCategories','test')
# a, b = d.split_indices(0.5)

# print(len(a), len(b))
# for a,b in d.data_list:
#    print(b)
