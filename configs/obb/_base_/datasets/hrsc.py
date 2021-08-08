dataset_type = 'HRSCDataset'
data_root = 'data/hrsc/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadOBBAnnotations', with_bbox=True,
         with_label=True, with_poly_as_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='OBBRandomFlip', h_flip_ratio=0.5, v_flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomOBBRotate', rotate_after_flip=True,
         angles=(0, 0), vert_rate=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='Mask2OBB', obb_type='obb'),
    dict(type='OBBDefaultFormatBundle'),
    dict(type='OBBCollect', keys=['img', 'gt_bboxes', 'gt_obboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipRotateAug',
        img_scale=[(1333, 800)],
        h_flip=False,
        v_flip=False,
        rotate=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='OBBRandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='RandomOBBRotate', rotate_after_flip=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='OBBCollect', keys=['img']),
        ])
]

# does evaluation while training
# uncomments it  when you need evaluate every epoch
# data = dict(
    # samples_per_gpu=2,
    # workers_per_gpu=4,
    # train=dict(
        # type=dataset_type,
        # imgset=data_root + 'ImageSets/train.txt',
        # ann_file=data_root + 'FullDataSet/Annotations/',
        # img_prefix=data_root + 'FullDataSet/AllImages/',
        # pipeline=train_pipeline),
    # val=dict(
        # type=dataset_type,
        # imgset=data_root + 'ImageSets/val.txt',
        # ann_file=data_root + 'FullDataSet/Annotations/',
        # img_prefix=data_root + 'FullDataSet/AllImages/',
        # pipeline=test_pipeline),
    # test=dict(
        # type=dataset_type,
        # imgset=data_root + 'ImageSets/val.txt',
        # ann_file=data_root + 'FullDataSet/Annotations/',
        # img_prefix=data_root + 'FullDataSet/AllImages/',
        # pipeline=test_pipeline))
# evaluation = dict(metric='mAP')

# disable evluation, only need train and test
# uncomments it when use trainval as train
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        imgset=data_root + 'ImageSets/trainval.txt',
        ann_file=data_root + 'FullDataSet/Annotations/',
        img_prefix=data_root + 'FullDataSet/AllImages/',
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        imgset=data_root + 'ImageSets/test.txt',
        ann_file=data_root + 'FullDataSet/Annotations/',
        img_prefix=data_root + 'FullDataSet/AllImages/',
        pipeline=test_pipeline))
evaluation = None
