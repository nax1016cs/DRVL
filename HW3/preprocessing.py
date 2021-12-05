import os, json, numpy as np
from tqdm import tqdm
from imantics import Mask, Image, Category, Dataset
import cv2
import argparse
import glob
import sys
sys.setrecursionlimit(30000)

parser=argparse.ArgumentParser()
parser.add_argument('--mode',type=str,default='train',help='train or valid')
parser.add_argument('--data_root',type=str,default='/eva_data/ming/DRVL/HW3/detection_data')
parser.add_argument('--mask_root',type=str,default='/eva_data/ming/DRVL/HW3/dataset/train')
parser.add_argument('--out_dir',type=str,default='/eva_data/ming/DRVL/HW3/result')
args=parser.parse_args()


dataset = Dataset('nuclei')

def convert(data_list,mask_list,mode,annot_dir):
    for idx,(file,path) in enumerate(zip(data_list,mask_list)):
        name=file.split('/')[-1]
        image = cv2.imread(file)[:,:,::-1]
        image = Image(image, id=idx) # 定义一个Image对象
        image.file_name=name
        image.path = file # 为Image对象添加coco标签格式的'path'属性

        pbar=tqdm(os.listdir(path))
        for index, i in enumerate(pbar):
            if i[-3:]!='png':
                continue
            pbar.set_description(i)
            mask_file = os.path.join(path, i)
            # name = i.split('_')[0]

            mask = cv2.imread(mask_file, 0)
            t = cv2.imread(file)
            if t.shape[:-1] != mask.shape:
                h, w, _ = t.shape
                mask = cv2.resize(mask, (w, h), cv2.INTER_CUBIC)

            mask = Mask(mask) # 定义一个Mask对象，并传入上面所定义的image对应的mask数组

            categ = 'nuclei'
            t = Category(categ) # 这里是定义Category对象
            t.id=1
            image.add(mask, t) # 将mask信息和类别信息传给image

        dataset.add(image) # 往dataset里添加图像以及gt信息
        print(f'{name} complete...')
        
    t = dataset.coco() # 将dataset转化为coco格式的，还可以转化为yolo等格式
    with open(f'{annot_dir}/instance_{mode}.json', 'w') as output_json_file: # 最后输出为json数据
        json.dump(t, output_json_file,indent=4)


if __name__=='__main__':
    data_path=os.path.join(args.data_root,args.mode)
    annot_dir=os.path.join(args.data_root,'annotations')
    if not os.path.isdir(annot_dir):
        os.mkdir(annot_dir)
    data_list=[data_name for data_name in glob.glob(f'{data_path}/*.png')]
    mask_list=[]

    for data_name in data_list:
        img_name=data_name.split('/')[-1]
        data_prefix=img_name[:-4]
        mask_path=os.path.join(args.mask_root,data_prefix,'masks')
        mask_list.append(mask_path)
    convert(data_list,mask_list,args.mode,annot_dir)