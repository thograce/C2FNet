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
Camouflaged object detection (COD) is a challenging task due to the low boundary contrast between the object and its surroundings. In addition, the appearance of camouflaged objects varies significantly, \eg, object size and shape, aggravating the difficulties of accurate COD. In this paper, we propose a novel Context-aware Cross-level Fusion Network (C2FNet) to address the challenging COD task.Specifically, we propose an Attention-induced Cross-level Fusion Module (ACFM) to integrate the multi-level features with informative attention coefficients. The fused features are then fed to the proposed Dual-branch Global Context Module (DGCM), which yields multi-scale feature representations for exploiting rich global context information. In C2FNet, the two modules are conducted on high-level features using a cascaded manner. Extensive experiments on three widely used benchmark datasets demonstrate that our C2FNet is an effective COD model and outperforms state-of-the-art models remarkably. 

### 2.2. Framework Overview


### 2.3. Qualitative Results


## 3. Proposed Baseline

### 3.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
a single NVIDIA Tesla P40 GPU of 24 GB Memory.

1. Configuring your environment (Prerequisites):
  
    Note that C2FNet is only tested on Ubuntu OS with the following environments. 
    It may work on other operating systems as well but we do not guarantee that it will.
    
    + Creating a virtual environment in terminal: `conda create -n C2FNet python=3.6`.
    
    + Installing necessary packages: PyTorch 1.3.1

1. Downloading necessary data:

    + downloading testing dataset and move it into `./data/TestDataset/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1QEGnP9O7HbN_2tH999O3HRIsErIVYalx/view?usp=sharing).
    
    + downloading training dataset and move it into `./data/TrainDataset/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1Kifp7I0n9dlWKXXNIbN7kgyokoRY4Yz7/view?usp=sharing).
    
    + downloading pretrained weights and move it into `checkpoints/C2FNet40/C2FNet-39.pth`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1owypj40dZjES8X0ex1QOHJox1NNBCgB-/view?usp=sharing).
    
    + downloading Res2Net weights [download link (Google Drive)](https://drive.google.com/file/d/1_1N-cx1UpRQo7Ybsjno1PAg4KE1T9e5J/view?usp=sharing).
   
1. Training Configuration:

    + Assigning your costumed path, like `--train_save` and `--train_path` in `MyTrain.py`.

1. Testing Configuration:

    + After you download all the pre-trained model and testing dataset, just run `MyTest.py` to generate the final prediction map: 
    replace your trained model directory (`--pth_path`).

### 3.2 Evaluating your trained model:

One-key evaluation is written in MATLAB code (revised from [link](https://github.com/DengPingFan/CODToolbox)), 
please follow this the instructions in `./eval/main.m` and just run it to generate the evaluation results in.

> pre-computed map can be found in [download link](https://drive.google.com/file/d/1LTE85A4CtQm3mJ9Dqbh_CT4_3tmcStXQ/view?usp=sharing).


## 4. Citation

Please cite our paper if you find the work useful: 

**[⬆ back to top](#0-preface)**
