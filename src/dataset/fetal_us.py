import os
import numpy as np
import cv2
import pandas as pd
import PIL

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class fetalhead_2d(Dataset):
    def __init__(self, root_dir, df1, df2, train=False, transform=None):
        super(Dataset, self).__init__()
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        if self.train == True:
            self.folder_path = self.root_dir + '/train/'
        else:
            self.folder_path = self.root_dir + '/test/'

        self.records = df1
        self.labels = df2.tolist()
        self.paths = [self.folder_path + filename for filename in self.records['image_name'].tolist()]

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(self.paths[index], cv2.IMREAD_COLOR) #load US images
        image = PIL.Image.fromarray(image)
        label = self.labels[index] #get label of ultrasound image
        label = torch.Tensor([label]) #convert type from numpy to torch
        # label = torch.as_tensor(label) #convert type from float to torch

        if self.transform is not None:
            image = self.transform(image)

        image = torch.from_numpy(image.numpy()[..., ::])

        return image, label