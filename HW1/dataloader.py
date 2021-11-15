import os
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image


def getData(mode):
    if mode == 'train':
        imgs = []
        labels = []
        f = open("./data/training_labels.txt")
        for line in f:
            img, label = line.strip().split(" ")
            imgs.append(img)
            labels.append(int(label.split(".")[0])-1)
        f.close()
        return imgs, labels

    elif mode == "test":
        imgs = []
        f = open("./data/testing_img_order.txt")
        for line in f:
            img = line.strip()
            imgs.append(img)
        f.close()
        return imgs


class BirdDataloader(data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        if mode == "train" or mode == "valid":
            self.img_name, self.label = getData(mode)
        elif mode == "test":
            self.img_name = getData(mode)

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        img = Image.open(self.root + self.img_name[index])
        if self.mode == "train":
            transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        elif self.mode == "test":
            transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        img = transform(img)
        if self.mode == "train":
            label = torch.tensor(int(self.label[index]))
            return img, label

        else:
            return img


def dataloader(mode, batch_size=8):
    path = f'./data/{mode}/'
    dataset = BirdDataloader(path, mode)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4)

if __name__ == '__main__':
    train_data = dataloader("train")
    print(len(train_data))
    test_data = dataloader("test")
    print(len(test_data))
