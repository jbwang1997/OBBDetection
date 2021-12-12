_base_ = [
    '../_base_/datasets/dota.py',
    '../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]

optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
# RetinaNet nms is slow in early stage, disable every epoch evaluation
evaluation = None

model = dict(
    type='S2ANet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='S2AHead',
        feat_channels=256,
        align_type='AlignConv',
        heads=[
            dict(
                type='ODMHead',
                num_classes=15,
                in_channels=256,
                feat_channels=256,
                stacked_convs=2,
                anchor_generator=dict(
                    type='Theta0AnchorGenerator',
                    scales=[4],
                    ratios=[1.0],
                    strides=[8, 16, 32, 64, 128]),
                bbox_coder=dict(
                    type='DeltaXYWHABBoxCoder',
                    target_means=(0., 0., 0., 0., 0.),
                    target_stds=(1., 1., 1., 1., 1.)),
                reg_decoded_bbox=False,
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            ),
            dict(
                type='ODMHead',
                num_classes=15,
                in_channels=256,
                feat_channels=256,
                stacked_convs=2,
                with_orconv=True,
                bbox_coder=dict(
                    type='DeltaXYWHABBoxCoder',
                    target_means=(0., 0., 0., 0., 0.),
                    target_stds=(1., 1., 1., 1., 1.)),
                reg_decoded_bbox=False,
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            )
        ]
    )
)

# training and testing settings
train_cfg = [
    dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
]
test_cfg = dict(
    skip_cls=[True, False],
    nms_pre=2000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='obb_nms', iou_thr=0.1),
    max_per_img=2000)

# # Use All images for training
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadOBBAnnotations', with_bbox=True,
#          with_label=True, with_poly_as_mask=True),
#     dict(type='LoadDOTASpecialInfo'),
#     dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
#     dict(type='OBBRandomFlip', h_flip_ratio=0.5, v_flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='RandomOBBRotate', rotate_after_flip=True,
#          angles=(0, 0), vert_rate=0.5, vert_cls=['roundabout', 'storage-tank']),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DOTASpecialIgnore', ignore_size=2),
#     dict(type='Mask2OBB', obb_type='obb'),
#     dict(type='OBBDefaultFormatBundle'),
#     dict(type='OBBCollect', keys=['img', 'gt_bboxes', 'gt_obboxes', 'gt_labels'])
# ]
#
# data = dict(
#     train=dict(
#         filter_empty_gt=False,
#         pipeline=train_pipeline
#     ),
# )
