from dataloader import *
from models import *
import time
import random
from sklearn.model_selection import KFold
import torch.utils.data as data
import os
import math
import torchvision.models as models
import configargparse


p = configargparse.ArgumentParser()
p.add_argument('--lr', type=float, default=1e-4,
               help='learning rate')
p.add_argument('--batch_size', type=int, default=24,
               help='batchsize')
p.add_argument('--momentum', type=float, default=0.9,
               help='momentum')
p.add_argument('--weight_decay', type=float, default=5e-3,
               help='weight_decay')
p.add_argument('--epochs', type=int, default=200,
               help='the total epochs to train the model')
opt = p.parse_args()


def save_checkpoint(path, model):
    state_dict = {"model_state_dict": model.state_dict()}
    torch.save(state_dict, path)


def load_checkpoint(path, model, device):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict["model_state_dict"])
    return model


def train(model, train_data, device):
    torch.manual_seed(0)
    epochs = opt.epochs
    criterion = nn.CrossEntropyLoss()
    best_test_acc = 0
    train_acc_record = []
    optimizer = optim.SGD(
        model.parameters(),
        lr=opt.lr,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay
    )
    model.to(device=device)
    for epoch in range(1, epochs+1):
        model.train()
        epoch_start_time = time.time()
        train_acc, train_loss = 0.0, 0.0
        for datas, labels in train_data:
            data = datas.to(device, dtype=torch.float)
            label = labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            prediction = model(data)
            batch_loss = criterion(prediction, label)
            batch_loss.backward()
            optimizer.step()
            train_acc += prediction.max(dim=1)[1].eq(label).sum().item()
            train_loss += batch_loss.item()
            training_accuracy = train_acc / len(train_data.dataset)
            training_loss = train_loss / len(train_data.dataset)
            train_acc_record.append(training_accuracy)
        checkpoint_path = 'resnext_ckpt'
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        save_checkpoint('resnext_ckpt/best_model_' + str(epoch) + '.pt',
                        model)
        print('epoch[\033[35m{:>4d}\033[00m/{:>4d}] {:.2f} sec(s) \033[32m \
            Train Acc:\033[00m {:.6f} Train Loss: {:.6f}'.format(
                epoch, epochs, time.time() - epoch_start_time,
                training_accuracy, training_loss))


def main():
    print(f'lr: {opt.lr}, batchsize: {opt.batch_size}, epochs: {opt.epochs}\
         weight_decay {opt.weight_decay}, momentum {opt.momentum}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = dataloader("train", opt.batch_size)
    model = ResNeXt(output=200)
    acc = train(model, train_data, device)

if __name__ == '__main__':
    main()
