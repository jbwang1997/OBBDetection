# Oriented R-CNN and Beyond

> Xingxing Xie, Gong Cheng, Jiabao Wang, Ke li, Xiwen Yao, Junwei Han,

## Introduction

![illustration](illustration.jpg)

Currently, two-stage oriented detectors are superior to single-stage competitors in accuracy, but the step of generating oriented proposals is still time-consuming, thus hindering the inference speed. This paper proposes an Oriented Region Proposal Network (Oriented RPN) to produce high-quality oriented proposals in a nearly cost-free manner. To this end, we present a novel representation manner of oriented objects, named midpoint offset representation, which avoids the complicated design of oriented proposal generation network. Built on Oriented RPN, we develop a simple yet effective oriented object detection framework, called Oriented R-CNN, which could accurately and efficiently detect oriented objects. Moreover, we extend Oriented R-CNN to the task of instance segmentation and realize a new proposal-based instance segmentation method, termed Oriented Mask R-CNN. Without bells and whistles, Oriented R-CNN achieves state-of-the-art accuracy on all 7 commonly-used oriented object detection datasets for the tasks of aerial object detection and scene text detection. More importantly, our method has the fastest speed among all detectors. For instance segmentation, Oriented Mask R-CNN also achieves the top results on the large-scale aerial instance segmentation dataset, named iSAID. We hope our methods could serve as solid baselines for oriented object detection and instance segmentation.

## Datasets

First, Users need to pre-process isaid dataset, Please follow the tutorial of [iSAID_Devkit](https://github.com/CAPTAIN-WHU/iSAID_Devkit) to split images and get the coco format json file.

## Results and models

| Backbone | Lr schd | ms | rr | mask mAP |                         Baidu Yun                         | Google Drive |
|:--------:|:-------:|:--:|:--:|:--------:|:---------------------------------------------------------:|:------------:|
|  R50-FPN |    1x   |  - |  - |   38.91  |  [uni7](https://pan.baidu.com/s/1holNWQ0MNpgGLs-8ctx-zQ)  |   [Model](https://drive.google.com/file/d/1JzxO6no_AnitZkEvdWkPRVwxtOV8KvGW/view?usp=sharing) |
| R101-FPN |    1x   |  - |  - |   39.22  |  [nxj3](https://pan.baidu.com/s/1AjbMhGLWNJ0aDuboexTx5A)  |   [Model](https://drive.google.com/file/d/1MlXX8MX4XpSVHzvPSIib4rq4Rkqfxs9O/view?usp=sharing)  |
| R152-FPN |    1x   |  - |  - |   40.15  |  [6qw8](https://pan.baidu.com/s/1vD6l4hX6VoxPx35qAq592A)  |   [Model](https://drive.google.com/file/d/1NJnuLiISFm6MH6_MjHt1JEnPHtc61B35/view?usp=sharing) |

## Citation

To be continue
