# OBBDetection

**news**: We are now updating OBBDetection to new vision based on MMdetection v2.10, which has more advanced models and more efficient features.

## introduction
OBBDetection is a open source oriented object detection toolbox based on the [MMdetection](https://github.com/open-mmlab/mmdetection).
We didn't change the codes in MMdetection, only add the extra codes in **obb** folders.
So the OBBDetection is accessible to all models and features in MMdetection.
![demo image](demo/dota_test_P0628_0001.png)

### Major features

- **MMdetection inheritance**

  OBBDetection didn't change any codes in MMdetection. Instead, we create a new folder, named **obb**, to collect the additive codes. All structures and features are same as MMdetection.

- **Support of multiple frameworks out of box**

  Except the common horizontal detection frameworks, the toolbox supports popular oriented detection frameworks *e.g.* Faster RCNN OBB, RoI Transformer, Gliding Vertex.

- **Flexible representation of boxes**

  The toolbox supports three type of bounding boxes, horizontal bounding boxes (HBB), oriented bounding boxes (OBB), and 4 point boxes (POLY). Each type of boxes can transforms to others directly.

- **Efficiency of training and testing big images**

  For dataset like DOTA, OBBDetection can directly merge patch results. So, it's available to validate big image dataset during training, and get the final results directly from testing.
  

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).

Supported backbones:
- [x] ResNet
- [x] ResNeXt
- [x] VGG
- [x] HRNet
- [x] RegNet
- [x] Res2Net

Supported oriented detection methods:
- [x] RPN OBB
- [x] [Faster R-CNN OBB](configs/obb/faster_rcnn_obb)
- [x] [RetinaNet OBB](configs/obb/retinanet_obb)
- [x] [Gliding Vertex](configs/obb/gliding_vertex)
- [x] [RoI Transformer](configs/obb/roi_transformer)

Supported horizontal detection methods:
- [x] [RPN](configs/rpn)
- [x] [Fast R-CNN](configs/fast_rcnn)
- [x] [Faster R-CNN](configs/faster_rcnn)
- [x] [Mask R-CNN](configs/mask_rcnn)
- [x] [Cascade R-CNN](configs/cascade_rcnn)
- [x] [Cascade Mask R-CNN](configs/cascade_rcnn)
- [x] [SSD](configs/ssd)
- [x] [RetinaNet](configs/retinanet)
- [x] [GHM](configs/ghm)
- [x] [Mask Scoring R-CNN](configs/ms_rcnn)
- [x] [Double-Head R-CNN](configs/double_heads)
- [x] [Hybrid Task Cascade](configs/htc)
- [x] [Libra R-CNN](configs/libra_rcnn)
- [x] [Guided Anchoring](configs/guided_anchoring)
- [x] [FCOS](configs/fcos)
- [x] [RepPoints](configs/reppoints)
- [x] [Foveabox](configs/foveabox)
- [x] [FreeAnchor](configs/free_anchor)
- [x] [NAS-FPN](configs/nas_fpn)
- [x] [ATSS](configs/atss)
- [x] [FSAF](configs/fsaf)
- [x] [PAFPN](configs/pafpn)
- [x] [Dynamic R-CNN](configs/dynamic_rcnn)
- [x] [PointRend](configs/point_rend)
- [x] [CARAFE](configs/carafe/README.md)
- [x] [DCNv2](configs/dcn/README.md)
- [x] [Group Normalization](configs/gn/README.md)
- [x] [Weight Standardization](configs/gn+ws/README.md)
- [x] [OHEM](configs/faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py)
- [x] [Soft-NMS](configs/faster_rcnn/faster_rcnn_r50_fpn_soft_nms_1x_coco.py)
- [x] [Generalized Attention](configs/empirical_attention/README.md)
- [x] [GCNet](configs/gcnet/README.md)
- [x] [Mixed Precision (FP16) Training](configs/fp16/README.md)
- [x] [InstaBoost](configs/instaboost/README.md)
- [x] [GRoIE](configs/groie/README.md)
- [x] [DetectoRS](configs/detectors/README.md)
- [x] [Generalized Focal Loss](configs/gfl/README.md)

## Installation

Please refer to [install.md](docs/install.md) for installation and dataset preparation.

## Get Started

### Oriented models training and testing

If you want to train and test a oriented model, please refer to [oriented_model_starting.md](docs/oriented_model_starting.md).

### How to use MMDetection

If you are not familiar with MMdetection, please see [getting_started.md](docs/getting_started.md) for the basic usage of MMDetection. There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), and [adding new modules](docs/tutorials/new_modules.md).

## Acknowledgement

This toolbox is based on [MMdetection](https://github.com/open-mmlab/mmdetection). If you use this toolbox or benchmark in your research, please cite the following information.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```