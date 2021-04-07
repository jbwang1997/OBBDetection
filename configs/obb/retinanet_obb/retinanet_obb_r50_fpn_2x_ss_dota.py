_base_ = './retinanet_obb_r50_fpn_1x_dota.py'
lr_config = dict(step=[16, 22])
total_epochs = 24
