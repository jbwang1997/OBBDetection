# [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://papers.nips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf)

Faster R-CNN towards oriented detection. Predicts the angles of objects in second stage.

## Introduction
```
@article{Ren_2017,
   title={Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
   year={2017},
   month={Jun},
}
```

## Results and models

### HRSC

| Backbone | Data | Lr schd | Ms | Rotate | box AP | Baidu Yun | Google Drive |
|:--------:|:----:|:-------:|:--:|:------:|:------:|:---------:|:------------:|
|  R50-FPN | HRSC |    3x   |  - |    -   |  80.9  |     -     |       -      |
|  R50-FPN | HRSC |    3x   |  - |    -   |  84.6  |     -     |       -      |
