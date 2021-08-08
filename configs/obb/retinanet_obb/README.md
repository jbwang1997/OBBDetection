# RetinaNet OBB

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

| Backbone |   Data  | Lr schd | Ms | Rotate | box AP | Log | Baidu Yun | Google Drive |
|:--------:|:-------:|:-------:|:--:|:------:|:------:|:---:|:---------:|:------------:|
| R50-FPN  | DOTA1.0 |    1x   |  - |    -   | 69.4%  | [log](logs/retinanet_obb_r50_fpn_1x_dota10.log)|[key:wqrb](https://pan.baidu.com/s/1cVF1jnt1ieEDk2eI9HVbRw)|[model](https://drive.google.com/file/d/1N1-POFkXfdTNlLGcRuj_nB2guWNNutZE/view?usp=sharing)|
