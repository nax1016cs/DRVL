# HW1
This is the first homework of VRDL course in NYCU, which is about to do bird species classification. 

## Hardware

- Ubuntu 18.04.5 LTS
- Intel(R) Core(TM) i9-7900X CPU @ 3.30GHz
- GeForce GTX 1080 Ti


## Installation

Clone the repository and enter the folder.
```sh
git clone https://github.com/nax1016cs/DRVL.git
cd DRVL && cd HW1
```
It is recommended to use Anaconda to install the needed environment.
```sh
conda env create  -f requirements.yml
conda activate vrdl_hw1
```

## Download the dataset

You can download the needed data through the link.
```sh
https://drive.google.com/drive/folders/1OKQHV0lRrBa-E9ncLx24l-uX1KKStfP_?usp=sharing
unzip data.zip
```

## Download the checkpoint
The trained checkpoint can be downloaded from the following link.
```sh
https://drive.google.com/file/d/1I9OXiGpOkbhNI7oUkYn8Rkr0HwsqHS-L/view?usp=sharing
```

## Structure
After finishing the steps above, the whole structure is shown as below.
```
hw1
│   dataloader.py
│   inference.py
│   train.py
│   models.py
│   requirements.txt
│   best_model_200.pt
└───data
    │   classes.txt
    │   testing_img_order.txt
    │   training_labels.txt
    └───train 
    │     │  0003.jpg
    │     │  0008.jpg
    │     │  ...
    │     
    └───test
          │  0001.jpg
          │  0002.jpg
          │  ...
          

```

## Training
You can train the model by the following command.The learning rate, epochs, momentum, weight decay and batch size can be specified by you.
```sh
python train.py --lr 1e-4 --epochs 200 --weight_decay 5e-3 --momentum 0.9 --batch_size 24
```


## Testing
Then you can infernece the model with following command.
```sh
python inference.py
```
