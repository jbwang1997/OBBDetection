# [FCOS: Fully Convolutional One-Stage Object Detection](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tian_FCOS_Fully_Convolutional_One-Stage_Object_Detection_ICCV_2019_paper.pdf)

## Introduction

The oriented form of fcos.

```
@article{tian2019fcos,
  title={FCOS: Fully Convolutional One-Stage Object Detection},
  author={Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  journal={arXiv preprint arXiv:1904.01355},
  year={2019}
}
```

In order to predict direction, I add an one channel convolution layer on the top of the regression features.
I concatenate the 4-dim bbox predtions and 1-dim theta predictions as the final obb predictions.
following original FCOS, I use the PolyIoULoss between the obb predictions and ground-truths as the loss of bboxes.

**note**: Cannot predict left, right, up, bottom targets with theta togather, because direction predictions cannot share same scale module with distance preditions.

## Results

| Backbone |   Data  | Lr schd | centerness on reg | center sampling | norm on bbox | box AP |
|:--------:|:-------:|:-------:|:-----------------:|:---------------:|:------------:|:------:|
|  R50-FPN | DOTA1.0 |    1x   |         -         |        -        |       -      |  72.1  |
