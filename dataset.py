import numpy as np
import torch
from torch.utils.data import Dataset


class DnRDataset(Dataset):

    def __init__(self, filename, transform=None):

        self.filename = filename
        self.transform = transform
        self.data = np.load(filename, allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ApplyOnKey(object):

    def __init__(self, on_key, func):
        self.on_key = on_key
        self.func = func

    def __call__(self, sample):
        data = sample[self.on_key]
        sample[self.on_key] = self.func(data)
        return sample

    def __repr__(self):
        return '{}(on_key={})'.format(self.__class__.__name__, self.on_key) + ' - ' + self.func.__repr__()