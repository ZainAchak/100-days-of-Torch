from more_itertools import first
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        # super().__init__()
        xy = np.loadtxt("./data/wine.csv", delimiter="," , dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # n_samples, 1
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

dataset = WineDataset()
# first_data = dataset[0]
# features, label = first_data
# print(features, label)

batch_size = 4
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)# , num_workers=1)
# dataiter = iter(dataloader) # Iterativly we'll call the data
# data = dataiter.next() # Everytime it'll return 4 samples 4(Features and Labels)
# features, labels = data
# print(features.shape, labels.shape)

# training Loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/batch_size)

print(f'Total Samples: {total_samples} \nTotal Iterations: {n_iterations}')

for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(dataloader):
        pass