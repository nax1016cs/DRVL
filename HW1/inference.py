from dataloader import *
from models import *
import time
import random
from sklearn.model_selection import KFold
import torch.utils.data as data
import os
import math
import torchvision.models as models


def save_checkpoint(path, model):
    state_dict = {"model_state_dict": model.state_dict()}
    torch.save(state_dict, path)


def load_checkpoint(path, model, device):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict["model_state_dict"])
    return model


def load_class():
    classes = []
    f = open("./data/classes.txt", 'r')
    for line in f:
        classes.append(line.strip())
    f.close()
    return classes


def load_test_data():
    data = []
    f = open("./data/testing_img_order.txt", 'r')
    for line in f:
        data.append(line.strip())
    f.close()
    return data


def test(model, test_data, device):
    f = open("answer.txt", 'w')
    test_img_name = load_test_data()
    classes = load_class()
    model.to(device=device)
    model.eval()
    for idx, datas in enumerate(test_data):
        data = datas.to(device, dtype=torch.float)
        prediction = model(data)
        prediction = prediction.max(dim=1)[1].sum().item()
        guess = classes[prediction]
        f.write(f'{test_img_name[idx]} {guess}\n')
    f.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = dataloader("test", 1)
    model = ResNeXt(output=200)
    model = load_checkpoint("./best_model_200.pt", model,
                            device)
    test(model, test_data, device)

if __name__ == '__main__':
    main()
