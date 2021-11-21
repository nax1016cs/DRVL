# HW2
This is the second homework of VRDL course in NYCU, which is about to do object detection on Street View House Numbers detection.

## Hardware

- Ubuntu 18.04.5 LTS
- Intel(R) Core(TM) i9-7900X CPU @ 3.30GHz
- GeForce GTX 1080 Ti


## Notebook link

[Notebook link](https://colab.research.google.com/drive/1ANLECrkBFCUwuojwRGmdB0jU9bGUCQPH?usp=sharing)

## Installation

Clone the repository and enter the folder.
```sh
git clone https://github.com/nax1016cs/DRVL.git
cd DRVL && cd HW2
```
It is recommended to use Anaconda to install the needed environment.
```sh
conda env create  -f environment.yml
conda activate vrdl_hw2
```

## Download the dataset

You can download the needed data through the link.
```sh
mkdir dataset && cd dataset && mkdir images && mkdir labels && cd images
https://drive.google.com/drive/folders/1aRWnNvirWHXXXpPPfcWlHQuzGJdXagoc
unzip train.zip
unzip test.zip
cd ../../ && python preprocessing.py
```

## Download the checkpoint
The trained checkpoint can be downloaded from the following link.
```sh
cd yolov5/
https://drive.google.com/file/d/1126LcdvHIF8EMFiJxuXAsm5xuaFuIV_9/view?usp=sharing
```

## Structure
After finishing the steps above, the whole structure is shown as below.
```
HW1
HW2
│   README.md
│   environment.yml
│   preprocessing.py
└───yolov5
│   │   config.yml
│   │   train.py
│   │   inference.py
│   │   best.pt
│   │   ....
│
└───dataset
	│
	│
	└───images
	│	│
	│	│ 
	│	│
	│	└───train 
	│	│     │  1.png
	│	│     │  2.png
	│	│     │  ...
	│	│     
	│	└───test
	│	│	  │  117.png
	│	│	  │  162.png
	│	│	  │  ...
	│	│	  
	│	└─── valid
	│		  │ XXX.png
	│		  │	YYY.png
	│		  │ ...
	│
	└───labels
		│	
		│ 
		│
		└───train 
		│     │  1.txt
		│     │  2.txt
		│     │  ...
		│     
		└───valid
		│     │  XXX.txt
		│     │  YYY.txt	
		│     │  ...

```

## Training
You can train the model by the following command.
```sh
python train.py --data config.yaml --weights yolov5x6.pt --img 640
```


## Testing
Then you can infernece the model with following command.
```sh
python inference.py --data config.yaml --weights  best.pt  --save-json  --task test --img 640
cp runs/val/exp/best_predictions.json answer.json
```
