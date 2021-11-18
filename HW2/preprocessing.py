'''
Code is reference from https://www.vitaarca.net/post/
tech/access_svhn_data_in_python/
'''

import numpy as np
import h5py
import os
import random
import cv2
import math
bbox_prop = ['height', 'left', 'top', 'width', 'label']
os.makedirs("dataset/labels", exist_ok=True)
os.makedirs("dataset/images/valid", exist_ok=True)
os.makedirs("dataset/labels/train", exist_ok=True)
os.makedirs("dataset/labels/valid", exist_ok=True)


def get_img_name(f, idx=0):
    img_name = ''.join(map(chr, f[names[idx][0]][()].flatten()))
    return(img_name)


def get_img_boxes(f, idx=0):
    meta = {key: [] for key in bbox_prop}
    box = f[bboxs[idx][0]]
    for key in box.keys():
        if box[key].shape[0] == 1:
            meta[key].append(int(box[key][0][0]))
        else:
            for i in range(box[key].shape[0]):
                meta[key].append(int(f[box[key][i][0]][()].item()))
    return meta


def mat_to_txt(f):
    print("Making txt file.")
    max_ = f['digitStruct/name'].shape[0]
    for i in range(max_):
        img_name = get_img_name(f, i)
        info = get_img_boxes(f, i)
        txt_file = './dataset/labels/train/' + img_name.replace('.png', '.txt')
        fp = open(txt_file, 'w')
        length = len(info['height'])
        print('./dataset/images/train/' + img_name)
        img = cv2.imread('./dataset/images/train/' + img_name)
        h, w, c = img.shape
        for idx in range(length):
            label = info['label'][idx]
            box_l = info['left'][idx]
            box_t = info['top'][idx]
            box_w = info['width'][idx]
            box_h = info['height'][idx]
            x_center = (box_l + box_w / 2) / w
            y_center = (box_t + box_h / 2) / h
            width = box_w / w
            height = box_h / h
            if label == 10:
                label = 0
            s = f'{label} {x_center} {y_center} {width} {height}'
            if idx != (length - 1):
                s += '\n'
            fp.write(s)
        fp.close()


def split_train_valid(ratio=0.1):
    entries = os.listdir("./dataset/images/train/")
    rand_idx = random.sample(range(len(entries)),
                             math.floor(len(entries)*ratio))
    for i in rand_idx:
        name = entries[i].split(".")[0]
        os.rename(f'dataset/images/train/{name}.png',
                  f'dataset/images/valid/{name}.png')
        os.rename(f'dataset/labels/train/{name}.txt',
                  f'dataset/labels/valid/{name}.txt')
    print("Train: ", len(entries) - len(rand_idx))
    print("Valid: ", len(rand_idx))


digit_file = os.path.join('./dataset/images/train/', 'digitStruct.mat')
f = h5py.File(digit_file, 'r')
names = f['digitStruct/name']
bboxs = f['digitStruct/bbox']
mat_to_txt(f)
split_train_valid()
