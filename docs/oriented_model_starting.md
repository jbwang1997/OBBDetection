# Oriented Model Starting

This page provides the basic tutorials for training and testing oriented models.

## Huge Image Demo

We provide a demo script to test a single huge image, like DOTA images.

```shell
python demo/huge_image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${SPLIT_CONFIG_FILE} \
	 [--device ${GPU_ID}] [--score-thr ${SCORE_THR}]
```

You can get a pretrained checkpoint from every models' [README.md](./../configs/obb/).

example
```shell
python demo/huge_image_demo.py demo/dota_demo.jpg configs/obb/oriented_rcnn/faster_rcnn_orpn_r50_fpn_1x_dota10.py \
	 ckpt/faster_rcnn_orpn_r50_fpn_1x_dota10_epoch12.pth BboxToolkit/tools/split_configs/dota1_0/ss_test.json
```

## Prepare dataset

All config files of oriented object datasets are put at `<OBBDet>/configs/obb/_base_/dataset`. Before training and testing, you need to add the dataset path to config files.

Especially, DOTA dataset need to be splitted and add the splitted dataset path to DOTA config files. We develop a script `img_split.py` at `<OBBDet>/BboxToolkit/tools/` to split images and generate patch labels.
The simplest way to use `img_split.py` is loading the json config in `BboxToolkit/tools/split_configs`. Please refer to [USAGE.md](https://github.com/jbwang1997/BboxToolkit/USAGE.md) for the details of `img_split.py`.

**example**
```shell
cd BboxToolkit/tools/
# modify the img_dirs, ann_dirs and save_dir in split_configs/dota1_0/ss_dota_train.json
python img_split.py --base_json split_configs/dota1_0/ss_dota_train.json
```

**note**: the `ss` and `ms` mean `single scale splitting` and `multiple scales`.

## training

The training process is same as MMdetection training process.
Please see [gettting_started.md](getting_started.md) for details.

```shell
# one GPU training
python tools/train.py ${CONFIG_FILE} [optional arguments]

# multiple GPUs traing
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

**note**: We test all oriented models on 1 GPU and with batch size of 2. the basic learning rate is 0.005. if your training batch size is different from ours, please remember to change the learing rate based the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677).

## testing

Most testing process is same as the MMdetection testing process.
Please see [gettting_started.md](getting_started.md) for details.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

If you use DOTA dataset, you should convert and merge bounding boxes from the patch coordinate system to the full image coordinate system.
We merge this function in the testing process of OBBDetection. It can straightly genreate full image resutls without running other program.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --format-only --options save_dir=${SAVE_DIR}
```
where, `${SAVE_DIR}` is the output path for full image results.
