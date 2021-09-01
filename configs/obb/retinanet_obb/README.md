# [RetinaNet OBB](https://arxiv.org/pdf/1708.02002.pdf)

## Introduction

The RetinaNet of orietned form. Deriectly treat the horizontal anchors as oriented anchors with 0 theta. Use OBBOverlaps to calculator IoU when assigning targets.

```
@inproceedings{lin2017focal,
  title={Focal loss for dense object detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2017}
}
```

## Results and Models

### DOTA1.0

| Backbone |   Data  | Lr schd | Ms | Rotate | box AP |                           Baidu Yun                          |                                         Google Drive                                        |
|:--------:|:-------:|:-------:|:--:|:------:|:------:|:------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|
|  R50-FPN | DOTA1.0 |    1x   |  - |    -   |  69.4  | [key: wqrb](https://pan.baidu.com/s/1cVF1jnt1ieEDk2eI9HVbRw) | [model](https://drive.google.com/file/d/1N1-POFkXfdTNlLGcRuj_nB2guWNNutZE/view?usp=sharing) |
|  R50-FPN | DOTA1.0 |    2x   |  - |    -   |  72.7  | [key: x41s](https://pan.baidu.com/s/1wwO3k_lbZpwoErfO9N6q5Q) |                                              -                                              |
| R101-FPN | DOTA1.0 |    1x   |  - |    -   |  70.6  | [key: f5k6](https://pan.baidu.com/s/1sNYVzMgWyPfjZBegVGhp4A) |                                              -                                              |
| R101-FPN | DOTA1.0 |    2x   |  - |    -   |  73.7  | [key: v8r9](https://pan.baidu.com/s/1dNhBwO4uMfanV8SrxeijCA) |                                              -                                              |

### HRSC

| Backbone | Data | Lr schd | Ms | Rotate | box AP |                           Baidu Yun                          | Google Drive |
|:--------:|:----:|:-------:|:--:|:------:|:------:|:------------------------------------------------------------:|:------------:|
|  R50-FPN | HRSC |    1x   |  - |    -   |  73.1  | [key: y5j8](https://pan.baidu.com/s/1b2aWAZ5oqKqG5GSThOYimA) |       -      |
|  R50-FPN | HRSC |    1x   |  - |    -   |  73.3  | [key: aew1](https://pan.baidu.com/s/1grCrAYO55Q38K0eQAGR5gw) |       -      |
