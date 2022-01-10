# HW4
This is the fourth homework of VRDL course in NYCU, which is about to do super resolution on the provided dataset.

## Hardware

- Ubuntu 18.04.5 LTS
- Intel(R) Core(TM) i9-7900X CPU @ 3.30GHz
- GeForce GTX 1080 Ti

## Installation

### Clone the repository and enter the folder.
```sh
git clone https://github.com/nax1016cs/DRVL.git
cd DRVL && cd HW4 
```
## Download the dataset
You can download the needed data through the link.
```sh
mkdir dataset && cd dataset
https://drive.google.com/file/d/1GL_Rh1N-WjrvF_-YOKOyvq0zrV6TF4hb/view
unzip datasets.zip 
cd ../DBPN-Pytorch/
```

### Prepare environment
Create a conda virtual environment and activate it.
```
conda env create -f environment.yml
conda activate vrdl_hw4
```



## Download the checkpoint
The trained checkpoint can be downloaded from the following link.
```sh
https://drive.google.com/file/d/1Kvb2zBF5T6kgPioCGq_cTRb-Fke0vS78/view?usp=sharing
```



## Train
```
python main.py --upscale_factor 3 --gpus 4 --data_dir ../dataset/ --hr_train_dataset training_hr_images/training_hr_images/  --patch_size 10
```
## Inference
```
python eval.py --upscale_factor 3 --gpus 4 --input_dir ../dataset/ --test_dataset testing_lr_images/testing_lr_images --output result --model best.pth --residual True
```

## Create submission
```
python rename.py 
cd result && zip -r answer.zip ./
```
