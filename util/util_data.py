import config
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image


class ColorAugmentation(object):
    def __init__(self):
        self.eig_vec = torch.Tensor([
            [0.4009, 0.7192, -0.5675],
            [-0.8140, -0.0045, -0.5808],
            [-0.4203, -0.6948, -0.5836]
        ])
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val))*0.1
        quality = torch.mm(self.eig_val*alpha, self.eig_vec)
        tensor = tensor + quality.view(3, 1, 1)
        return tensor


def unpickle(file):
    with open(file, 'rb') as fo:
        result = pickle.load(fo, encoding='bytes')
    return result


def data_load(purpose="Train", size=0.1):
    data = []
    target = []

    if purpose == "Train":
        batches = list(range(1, 6))
    else:
        batches = ['test']

    for batch in batches:
        file = f'data/cifar-10-batches-py/data_batch_{batch}'
        raw = unpickle(file)
        target.extend(raw[b'labels'])
        data.extend(raw[b"data"])
    data = np.array(data)
    data = data.reshape(data.shape[0], 3, 32, 32)
    data = np.transpose(data, (0, 3, 2, 1))

    if purpose == "Train":
        return train_test_split(data, target, test_size=size)

    return data, target


class CIFAR10(Dataset):
    def __init__(self, data, target, transform=None):
        super(CIFAR10, self).__init__()
        self.transform = transform
        self.data = data
        self.target = torch.Tensor(target).to(config.device)

    def __getitem__(self, index):
        img = Image.fromarray(np.uint8(self.data[index]))
        img = self.transform(img) if self.transform is not None else img
        img = torch.Tensor(img).to(config.device)
        return img, self.target[index]

    def __len__(self):
        return len(self.data)
