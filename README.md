# C2FNet: Context-aware Cross-level Fusion Network for Camouflaged Object Detection (IJCAI 2021)

> **Authors:** 
> Yujia Sun,
> Geng Chen,
> Tao Zhou,
> Yi Zhang,
> and Nian Liu.

## 1. Preface

- This repository provides code for "_**Context-aware Cross-level Fusion Network for Camouflaged Object Detection**_" IJCAI-2021. 

## 2. Overview

### 2.1. Introduction


### 2.2. Framework Overview


### 2.3. Qualitative Results


## 3. Proposed Baseline

### 3.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
a single GeForce RTX TITAN GPU of 24 GB Memory.

> Note that our model also supports low memory GPU, which means you can lower the batch size


1. Configuring your environment (Prerequisites):
   
    Note that PraNet is only tested on Ubuntu OS with the following environments. 
    It may work on other operating systems as well but we do not guarantee that it will.
    
    + Creating a virtual environment in terminal: `conda create -n SINet python=3.6`.
    
    + Installing necessary packages: PyTorch 1.1

1. Downloading necessary data:

    + downloading testing dataset and move it into `./data/TestDataset/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/17FQbbvnKhNYbw8qsL7msx1tSP0cMV0no/view?usp=sharing).
    
    + downloading training dataset and move it into `./data/TrainDataset/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view?usp=sharing).
    
    + downloading pretrained weights and move it into `snapshots/C2FNet40/C2FNet-39.pth`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1owypj40dZjES8X0ex1QOHJox1NNBCgB-/view?usp=sharing).
    
    + downloading Res2Net weights [download link (Google Drive)](https://drive.google.com/file/d/1_1N-cx1UpRQo7Ybsjno1PAg4KE1T9e5J/view?usp=sharing).
   
1. Training Configuration:

    + Assigning your costumed path, like `--train_save` and `--train_path` in `MyTrain.py`.
    
    + Just enjoy it!

1. Testing Configuration:

    + After you download all the pre-trained model and testing dataset, just run `MyTest.py` to generate the final prediction map: 
    replace your trained model directory (`--pth_path`).
    
    + Just enjoy it!

### 3.2 Evaluating your trained model:

One-key evaluation is written in MATLAB code (revised from [link](https://github.com/DengPingFan/CODToolbox)), 
please follow this the instructions in `./eval/main.m` and just run it to generate the evaluation results in.

> pre-computed map can be found in [download link](https://drive.google.com/file/d/1LTE85A4CtQm3mJ9Dqbh_CT4_3tmcStXQ/view?usp=sharing).


## 4. Citation

Please cite our paper if you find the work useful: 

**[⬆ back to top](#0-preface)**
