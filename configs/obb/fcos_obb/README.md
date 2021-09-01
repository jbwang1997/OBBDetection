# FCOS: Fully Convolutional One-Stage Object Detection

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

To be continue!!!
