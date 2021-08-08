_base_ = './gliding_vertex_r50_fpn_1x_dota10.py'

# model
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
