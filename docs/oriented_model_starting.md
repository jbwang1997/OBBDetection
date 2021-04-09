# Oriented Model Starting

This page provides the basic tutorials for traing and testing oriented models.

## Prepare dataset

For most of the dataset, you need to change the dataset paths in config files before training and testing.
The default oriented setting are collected at `configs/obb/_base_/dataset/`.

DOTA is a special dataset, where the images are too big for direct training and testing. We first split DOTA images into small patches using **img_split.py** in [BboxToolkit](https://github.com/jbwang1997/BboxToolkit), and add the splitted dataset path in the DOTA dataset config file.

The usage of **img_split.py** can refer to [USAGE.md](https://github.com/jbwang1997/BboxToolkit/USAGE.md). The simplest way is changing the default setting in `BboxToolkit/tools/split_configs/`.

**example**
```shell
cd BboxToolkit/tools/
# change the img_dirs ann_dirs and save_dir in ss_dota_train.json
python img_split.py --base_json split_configs/ss_dota_train.json
```
Add the splitted dataset path in DOTA dataset config file.

## training

The training process is similar with MMdetection.
Please see [gettting_started.md](getting_started.md) for details.

```shell
# one GPU training
python tools/train.py ${CONFIG_FILE} [optional arguments]

# multiple GPUs traing
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

## testing

Most testing process is similar with MMdetection.
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