_base_ = './faster_rcnn_roitrans_r50_fpn_3x_hrsc.py'

# model
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
