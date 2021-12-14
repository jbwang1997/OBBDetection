# [Align Deep Features for Oriented Object Detection](https://arxiv.org/pdf/2008.09397.pdf)

## Introduction
The past decade has witnessed significant progress on detecting objects in aerial images that are often distributed with large scale variations and arbitrary orientations. However most of existing methods rely on heuristically defined anchors with different scales, angles and aspect ratios and usually suffer from severe misalignment between anchor boxes and axis-aligned convolutional features, which leads to the common inconsistency between the classification score and localization accuracy. To address this issue, we propose a Single-shot Alignment Network (S2A-Net) consisting of two modules: a Feature Alignment Module (FAM) and an Oriented Detection Module (ODM). The FAM can generate high-quality anchors with an Anchor Refinement Network and adaptively align the convolutional features according to the corresponding anchor boxes with a novel Alignment Convolution. The ODM first adopts active rotating filters to encode the orientation information and then produces orientation-sensitive and orientation-invariant features to alleviate the inconsistency between classification score and localization accuracy. Besides, we further explore the approach to detect objects in large-size images, which leads to a better speed-accuracy trade-off. Extensive experiments demonstrate that our method can achieve state-of-the-art performance on two commonly used aerial objects datasets (i.e., DOTA and HRSC2016) while keeping high efficiency.
```
@article{han2021align,  
  author={J. {Han} and J. {Ding} and J. {Li} and G. -S. {Xia}},  
  journal={IEEE Transactions on Geoscience and Remote Sensing},   
  title={Align Deep Features for Oriented Object Detection},   
  year={2021}, 
  pages={1-11},  
  doi={10.1109/TGRS.2021.3062048}}

```

## Results and Models

### DOTA1.0

| Backbone |   Data  | Lr schd | Ms | Rotate | box AP |                           Baidu Yun                          |                                         Google Drive                                        |
|:--------:|:-------:|:-------:|:--:|:------:|:------:|:------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|
|  R50-FPN | DOTA1.0 |    1x   |  - |    -   |  74.0  | [key: gemn](https://pan.baidu.com/s/11P3BZEfBsiJUCEXT4SNeYg) |                                              -                                              |
|  R50-FPN | DOTA1.0 |    2x   |  - |    -   |    -   |                                                              |                                              -                                              |
| R101-FPN | DOTA1.0 |    1x   |  - |    -   |    -   |                                                              |                                              -                                              |
| R101-FPN | DOTA1.0 |    2x   |  - |    -   |    -   |                                                              |                                              -                                              |
