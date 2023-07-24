import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):

    def __init__(self, transform=None):
        xy = np.loadtxt(r'C:\Users\Szkolenie\Documents\GitHub\LSTM-quaternion-sequence-extrapolation\Tests\11_dataloader\wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.n_samples = xy.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample =  self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

class ToTensor():
    def __call__(self, sample):
        inputs, lables = sample
        return torch.from_numpy(inputs), torch.from_numpy(lables)

class MulTransform():
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, labels = sample
        inputs *= self.factor
        return inputs, labels

dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))