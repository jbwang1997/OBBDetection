_base_ = './retinanet_obb_r50_fpn_1x_dota10.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
