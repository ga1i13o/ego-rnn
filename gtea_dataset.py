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
# directory containing the x-flows frames
FLOW_X_FOLDER = "flow_x_processed"
# directory containing the y-flows frames
FLOW_Y_FOLDER = "flow_y_processed"
# directory containing the rgb frames
FRAME_FOLDER = "processed_frames2"


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # functions that loads an image as an rgb pil object
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def flow_pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # functions that loads an image as a gray-scale pil object
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


class GTEA61(VisionDataset):
    # this class inherites from VisionDataset and represents the rgb frames of the dataset
    def __init__(self, root, split='train', seq_len=16, transform=None, target_transform=None, label_map=None):
        super(GTEA61, self).__init__(root, transform=transform, target_transform=target_transform)
        self.datadir = root
        # split indicates whether we should load the train or test split
        self.split = split
        # seq len tells us how many frames for each video we are going to consider
        # frames will be taken uniformly spaced
        self.seq_len = seq_len
        self.label_map = label_map
        if label_map is None:
            # if the label map dictionary is not provided, we are going to build it
            self.label_map = {}
        # videos is a list containing for each video, its path where you can find all its frames
        self.videos = []
        # labels[i] contains the class ID of the i-th video
        self.labels = []
        # n_frames[i] contains the number of frames available for i-th video
        self.n_frames = []

        # we expect datadir to be GTEA61, so we add FRAME_FOLDER to get to the frames
        frame_dir = os.path.join(self.datadir, FRAME_FOLDER)
        users = os.listdir(frame_dir)
        if len(users) != 4:
            raise FileNotFoundError("you specified the wrong directory")
        if TEST_USER not in users:
            raise FileNotFoundError("S2 folder not found")
        if self.split == 'test':
            folders = [users[users.index(TEST_USER)]]
        else:
            users.remove(TEST_USER)
            folders = users

        # folders is a list that contains either :
        #   - 1 element -> the path of the folder of the user S2 if split == 'test'
        #   - 3 elements -> the paths of the folders for S1,S3,S4 if split == 'train'

        if label_map is None:
            # now we build the label map; we take folders[0] just to get all class names
            # since it is GUARANTEED that all users have same classes
            classes = os.listdir(os.path.join(frame_dir, folders[0]))
            self.label_map = {act: i for i, act in enumerate(classes)}
        for user in folders:
            user_dir = os.path.join(frame_dir, user)
            # user dir it's gonna be ../GTEA61/processed_frames2/S1 or any other user
            for action in os.listdir(user_dir):
                action_dir = os.path.join(user_dir, action)
                # inside an action dir we can have 1 or more videos
                for element in os.listdir(action_dir):
                    # we add rgb to the path since there is an additional folder inside S1/1/rgb
                    # before the frames
                    frames = os.path.join(action_dir, element, "rgb")
                    # we append in videos the path
                    self.videos.append(frames)
                    # in labels the label, using the label map
                    self.labels.append(self.label_map[action])
                    # in frames its length in number of frames
                    self.n_frames.append(len(os.listdir(frames)))

    def __getitem__(self, index):
        # firstly we retrieve the video path, label and num of frames
        vid = self.videos[index]
        label = self.labels[index]
        length = self.n_frames[index]
        if self.transform is not None:
            # this is needed to randomize the parameters of the random transformations
            self.transform.randomize_parameters()

        # sort the list of frames since the name is like rgb002.png
        # so we use the last number as an ordering
        frames = np.array(sorted(os.listdir(vid)))
        # now we take seq_len equally spaced frames between 0 and length
        # linspace with the option int will give us the indices to take
        select_indices = np.linspace(0, length, self.seq_len, endpoint=False, dtype=int)
        # we then select the frames using numpy fancy indexing
        # note that the numpy arrays are arrays of strings, containing the file names
        # nevertheless, numpy will work with string arrays as well
        select_frames = frames[select_indices]
        # append to each file its path
        select_files = [os.path.join(vid, frame) for frame in select_frames]
        # use pil_loader to get pil objects
        sequence = [pil_loader(file) for file in select_files]
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            sequence = [self.transform(image) for image in sequence]
        # now, since the last transformation applied is always toTensor(),
        # we have in sequence a list of tensor, so we use stack along dimension 0
        # to create a tensor with one more dimension that contains them all
        sequence = torch.stack(sequence, 0)

        return sequence, label

    def __len__(self):
        return len(self.videos)


class GTEA61_flow(VisionDataset):
    # this class inherites from VisionDataset and represents the rgb frames of the dataset
    def __init__(self, root, split='train', seq_len=5, transform=None, target_transform=None, label_map=None):
        super(GTEA61_flow, self).__init__(root, transform=transform, target_transform=target_transform)
        # we expect datadir to be ../GTEA61
        self.datadir = root
        # split indicates whether we should load the train or test split
        self.split = split
        # seq len here tells us how many optical frames for each video
        # we are going to consider; note that now
        # frames will be sequential and not uniformly spaced
        self.seq_len = seq_len
        self.label_map = label_map
        if label_map is None:
            # if the label map dictionary is not provided, we are going to build it
            self.label_map = {}
        # x_frames is a list containing for each flow video, its path, where you can find all its frames
        # it will contain the ones under flow_x_processed
        self.x_frames = []
        # y_frames is the same as x_frames, but contains the ones under flow_y_processed
        self.y_frames = []
        # labels[i] contains the class ID of the i-th video
        self.labels = []
        # n_frames[i] contains the number of frames available for i-th video
        self.n_frames = []

        # we expect datadir to be GTEA61, so we add the flow folder to get to the flow frames
        flow_dir = os.path.join(self.datadir, FLOW_X_FOLDER)
        users = os.listdir(flow_dir)
        if len(users) != 4:
            raise FileNotFoundError("you specified the wrong directory")
        if TEST_USER not in users:
            raise FileNotFoundError("S2 folder not found")
        if self.split == 'test':
            folders = [users[users.index(TEST_USER)]]
        else:
            users.remove(TEST_USER)
            folders = users

        # folders is a list that contains either :
        #   - 1 element -> the path of the folder of the user S2 if split == 'test'
        #   - 3 elements -> the paths of the folders for S1,S3,S4 if split == 'train'

        if label_map is None:
            # now we build the label map; we take folders[0] just to get all class names
            # since it is GUARANTEED that all users have same classes
            classes = os.listdir(os.path.join(flow_dir, folders[0]))
            self.label_map = {act: i for i, act in enumerate(classes)}

        for user in folders:
            # user dir it's gonna be ../GTEA61/flow_x_processed/S1 or any other user
            user_dir = os.path.join(flow_dir, user)
            for action in os.listdir(user_dir):
                # inside an action dir we can have 1 or more videos
                action_dir = os.path.join(user_dir, action)
                for element in os.listdir(action_dir):
                    frames = os.path.join(action_dir, element)
                    # we put in x_frames the path to the folder with all the flow frames
                    self.x_frames.append(frames)
                    # the path for the y_frames is the same as x, except that we replace
                    # flow_x_processed with flow_y_processed in the path
                    # it is GUARANTEED that for each action we have the same number
                    # of x and y frames
                    self.y_frames.append(frames.replace(FLOW_X_FOLDER, FLOW_Y_FOLDER))
                    # put the label in label using the label map dictionary
                    self.labels.append(self.label_map[action])
                    # put here the number of flow frames
                    self.n_frames.append(len(os.listdir(frames)))

    def __getitem__(self, index):
        # get the paths of the x video, y, label and length
        vid_x = self.x_frames[index]
        vid_y = self.y_frames[index]
        label = self.labels[index]
        length = self.n_frames[index]
        # needed to randomize the parameters of the custom transformations
        self.transform.randomize_parameters()

        # sort the list of frames since the name is like flow_x_002.png
        # so we use the last number as an ordering
        frames_x = np.array(sorted(os.listdir(vid_x)))
        # do the same for y
        frames_y = np.array(sorted(os.listdir(vid_y)))
        if self.split == 'train':
            # if we are training, we take a random starting frame
            startFrame = random.randint(0, length - self.seq_len)
        else:
            # if we are testing, we take a centered interval
            startFrame = np.ceil((length - self.seq_len) / 2)
        # the frames will be sequential, so the select indices are
        # from startFrame to starFrame + seq_len
        select_indices = startFrame + np.arange(0, self.seq_len)
        # we then select the frames using numpy fancy indexing
        # note that the numpy arrays are arrays of strings, containing the file names
        # nevertheless, numpy will work with string arrays as well
        select_x_frames = frames_x[select_indices]
        select_y_frames = frames_y[select_indices]
        # this will position the elements of select_x_frames and select_y_frames
        # alternatively in a numpy array. select_frames is gonna be like x_frame, y_frame, x_frame, y _frame...
        # remember that these array contain the file names of the frames
        select_frames = np.ravel(np.column_stack((select_x_frames, select_y_frames)))

        # append to each file the root path. for the root path we use the one for
        # x frames, and then replace it with a y for the y frames
        # remember that x frames are in even positions, and y in odd positions
        select_files = [os.path.join(vid_x, frame) for frame in select_frames]
        select_files[1::2] = [y_files.replace('x','y') for y_files in select_files[1::2]]
        # create pil objects
        sequence = [flow_pil_loader(file) for file in select_files]
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            # we apply different transformations for x and y frames
            # specifically, inv=True will create the negative image for x frames
            sequence[::2] = [self.transform(image, inv=True, flow=True) for image in sequence[::2]]
            sequence[1::2] = [self.transform(image, inv=False, flow=True) for image in sequence[1::2]]
        # now, since the last transformation applied is always toTensor(),
        # we have in 'sequence' a list of tensors, so we use stack along dimension 0
        # to create a tensor with one more dimension that contains them all
        # then we apply squeeze along the 1 dimension, because the images are gray-scale,
        # so there is only one channel and we eliminate that dimension
        sequence = torch.stack(sequence, 0).squeeze(1)

        return sequence, label

    def __len__(self):
        return len(self.x_frames)


class GTEA61_2Stream(VisionDataset):
    # this class inherites from VisionDataset and represents both rgb and flow frames of the dataset
    # it does so by wrapping together an instance of GTEA61 for the rgb frames
    # and an instance of GTEA61_flow for the flow frames
    def __init__(self, root, split='train', seq_len=7, stack_size=5, transform=None, target_transform=None):
        super(GTEA61_2Stream, self).__init__(root, transform=transform, target_transform=target_transform)
        # we expect datadir to be ../GTEA61
        self.datadir = root
        # split indicates whether we should load the train or test split
        self.split = split
        # seq len is the number of rgb frames. they will be uniformly spaced
        self.seq_len = seq_len
        # stack size is the number of flow frames. they will be sequential
        self.stack_size = stack_size

        # now we check that we are in the right directory
        frame_dir = os.path.join(self.datadir, FRAME_FOLDER)
        users = os.listdir(frame_dir)
        if len(users) != 4:
            raise FileNotFoundError("you specified the wrong directory")
        if TEST_USER not in users:
            raise FileNotFoundError("S2 folder not found")
        if self.split == 'test':
            folders = [users[users.index(TEST_USER)]]
        else:
            users.remove(TEST_USER)
            folders = users
        # now we build a label map dictionary and we pass it to the instances of GTEA and GTEA_flow
        classes = os.listdir(os.path.join(frame_dir, folders[0]))
        self.label_map = {act: i for i, act in enumerate(classes)}
        # instance the rgb dataset
        self.frame_dataset = GTEA61(self.datadir, split=self.split, seq_len=self.seq_len,
                                    transform=self.transform, label_map=self.label_map)
        # instance the flow dataset
        self.flow_dataset = GTEA61_flow(self.datadir, split=self.split, seq_len=self.stack_size,
                                        transform=self.transform, label_map=self.label_map)

    def __getitem__(self, index):
        # to retrieve an item, we just ask the instances of
        # rgb and flow dataset to do it
        # then we return both the tensors, and the label
        frame_seq, label = self.frame_dataset.__getitem__(index)
        flow_seq, _ = self.flow_dataset.__getitem__(index)
        return flow_seq, frame_seq, label

    def __len__(self):
        return self.frame_dataset.__len__()
