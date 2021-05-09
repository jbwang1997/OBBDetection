# Oriented Model Starting

This page provides the basic tutorials for training and testing oriented models.

## Prepare dataset

All config files of default oriented object datasets are put at `<OBBDet>/configs/obb_base_/dataset`. Before training and testing, you need to add the dataset path in config files.

**note**

The images of DOTA dataset need to be splitted because they are too big to train and test directly. We developed a script **img_split.py** at `<OBBDet>/BboxToolkit/tools/` to split images and generate labels of patches.

The simplest way to use **img_split.py** is loading the default json config in `BboxToolkit/tools/split_configs`. Please refer to [USAGE.md](https://github.com/jbwang1997/BboxToolkit/USAGE.md) for the details of **img_split.py**.

**example**
```shell
cd BboxToolkit/tools/
# modify the img_dirs ann_dirs and save_dir in ss_dota_train.json
python img_split.py --base_json split_configs/ss_dota_train.json
```

**note**: the 'ss' and 'ms' mean 'single scale' and 'multiple scales'.

## training

The training process is same as MMdetection training process.
Please see [gettting_started.md](getting_started.md) for details.

```shell
# one GPU training
python tools/train.py ${CONFIG_FILE} [optional arguments]

# multiple GPUs traing
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

**note**: We test all oriented model on 1 GPU and with batch size of 2. the basic learning rate is 0.005. if your training batch size is different from ours, please remember to change the learing rate based the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677).

## testing

Most testing process is same as MMdetection testing process.
Please see [gettting_started.md](getting_started.md) for details.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

For DOTA dataset, The output of model is only the results on patches. We need to merge the patch results into full image as final submission.

In OBBDetection, we write the merging function in DOTA dataset. The program can directly output the full image results by command:

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --format-only --options save_dir=${SAVE_DIR}
```
where, `${SAVE_DIR}` is the output path for full image results.
