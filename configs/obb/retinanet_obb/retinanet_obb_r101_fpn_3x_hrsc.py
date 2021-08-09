_base_ = './retinanet_obb_r50_fpn_3x_hrsc.py'

# model
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
