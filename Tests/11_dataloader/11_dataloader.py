import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):

    def __init__(self):
        xy = np.loadtxt(r'C:\Users\Szkolenie\Documents\GitHub\LSTM-quaternion-sequence-extrapolation\Tests\11_dataloader\wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
dataset = WineDataset()
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

# >>> Taking the next item <<<
# data_iter = iter(dataloader)
# data = next(data_iter)
# features, labels = data
# print(features, labels)

# Dummy training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward
        if (i+1) % 5 == 0:
            print(f'epoch: {epoch+1}/{num_epochs}, step: {i+1}/{n_iterations}, inputs: {inputs.shape}')
        # backward

        # update
