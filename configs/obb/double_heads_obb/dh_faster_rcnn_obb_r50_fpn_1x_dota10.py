_base_ = '../faster_rcnn_obb/faster_rcnn_obb_r50_fpn_1x_dota10.py'
model = dict(
    roi_head=dict(
        type='OBBDoubleHeadRoIHead',
        reg_roi_scale_factor=1.2,
        bbox_head=dict(
            _delete_=True,
            type='OBBDoubleConvFCBBoxHead',
            start_bbox_type='hbb',
            end_bbox_type='obb',
            num_convs=4,
            num_fcs=2,
            in_channels=256,
            conv_out_channels=1024,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=15,
            bbox_coder=dict(
                type='HBB2OBBDeltaXYWHTCoder',
                target_means=[0., 0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2, 1]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=2.0))))
