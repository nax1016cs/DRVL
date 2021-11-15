'''
Code is reference from https://www.vitaarca.net/post/tech/access_svhn_data_in_python/ 
'''

import numpy as np
import h5py
import os 
import random
import cv2 

def get_img_name(f, idx=0):
    img_name = ''.join(map(chr, f[names[idx][0]][()].flatten()))
    return(img_name)

bbox_prop = ['height', 'left', 'top', 'width', 'label']
def get_img_boxes(f, idx=0):
    meta = { key : [] for key in bbox_prop}
    box = f[bboxs[idx][0]]
    for key in box.keys():
        if box[key].shape[0] == 1:
            meta[key].append(int(box[key][0][0]))
        else:
            for i in range(box[key].shape[0]):
                meta[key].append(int(f[box[key][i][0]][()].item()))
    return meta

digit_file = os.path.join('./images/train/', 'digitStruct.mat')
f = h5py.File(digit_file, 'r')
max_ = f['digitStruct/name'].shape[0]
names = f['digitStruct/name']
bboxs = f['digitStruct/bbox']


for i in range(max_):
    img_name = get_img_name(f, i)
    info = get_img_boxes(f, i)
    fp = open('labels/train/' + img_name.replace('.png','.txt'), 'w')
    length = len(info['height'])
    img = cv2.imread( 'images/train/' + img_name)
    h, w, c = img.shape
    for idx in range(length):
        label = info['label'][idx]
        if label == 10:
            label = 0
        box_l = info['left'][idx]
        box_t = info['top'][idx]
        box_w = info['width'][idx]
        box_h = info['height'][idx]
        x_center = (box_l + box_w / 2) / w
        y_center = (box_t + box_h / 2) / h
        bbox_width = box_w / w
        bbox_height = box_h / h
        s = str(label)+' '+str(x_center)+' '+str(y_center)+' '+str(bbox_width)+' '+str(bbox_height)
        if idx != (length - 1):
            s += '\n'
        fp.write(s)
    fp.close()