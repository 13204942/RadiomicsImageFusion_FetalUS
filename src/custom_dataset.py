import numpy as np
import torch
from torch.utils.data import Dataset


class RadiomicDataset(Dataset):
    def __init__(self, train, labels, transform=None):

        self.train = train
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def get_data(self):
        return self.train

    def __getitem__(self, idx):
        x = torch.tensor(self.train[idx]).unsqueeze(0)
        y = torch.tensor(self.labels[idx]).unsqueeze(0)

        if self.transform:
            x, y = self.transform(x), self.transform(y)

        return x.float(), y.float()