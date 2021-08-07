_base_ = './retinanet_obb_r101_fpn_1x_dota10.py'

lr_config = dict(step=[16, 22])
total_epochs = 24
