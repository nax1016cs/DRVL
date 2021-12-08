# HW3
This is the second homework of VRDL course in NYCU, which is about to do object detection on Street View House Numbers detection.

## Hardware

- Ubuntu 18.04.5 LTS
- Intel(R) Core(TM) i9-7900X CPU @ 3.30GHz
- GeForce GTX 1080 Ti


## Installation

### Clone the repository and enter the folder.
```sh
git clone https://github.com/nax1016cs/DRVL.git
cd DRVL && cd HW3
```
### Prepare environment
1. Create a conda virtual environment and activate it.
```
conda create -n vrdl_hw3 python=3.7 -y
conda activate vrdl_hw3
```
2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,
```
conda install pytorch torchvision -c pytorch
```
Note: Make sure that your compilation CUDA version and runtime CUDA version match. You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

`E.g.1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install PyTorch 1.5, you need to install the prebuilt PyTorch with CUDA 10.1.
```
conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
```
`E.g. 2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install PyTorch 1.3.1., you need to install the prebuilt PyTorch with CUDA 9.2.
```
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
```
If you build PyTorch from source instead of installing the prebuilt package, you can use more CUDA versions such as 9.0.

## Install MMDetection
```
pip install openmim
mim install mmdet
```

## Install pycocotools
```
pip install git+git://github.com/waspinator/pycococreator.git@0.2.0
```

## Get and preprocess data
```
https://drive.google.com/file/d/1vBgslO4A3rrBBReQhLK8reS_r44ESSlK/view?usp=sharing
unzip dataset.zip
```

## Preprocess data
You can use the existing nuclei.json file, or regenerate it with following command.
```
python create_coco_json.py
```
## Download checkpoint
```
cd mmdetection
https://drive.google.com/file/d/1kJfx5BdxIM1X8qZHilwhYh3oA4hC4iKM/view?usp=sharing
```
## Change the data root in config.py
In the 8-th line of config.py, please change it to the correct directory.
```
data_root = 'path/to/dir'
```

## Training
Download the pre-trained model and train with it.
```
mkdir ckpt && cd ckpt
wget https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_2x_coco/mask_rcnn_x101_32x4d_fpn_2x_coco_bbox_mAP-0.422__segm_mAP-0.378_20200506_004702-faef898c.pth
cd .. && python tools/train.py config.py
```


## Inference
```
python tools/test.py config.py best.pth --eval bbox segm --options "jsonfile_prefix=./results"
mv results.segm.json answer.json
```
