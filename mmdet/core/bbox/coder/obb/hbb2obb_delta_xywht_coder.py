import numpy as np
import torch

from ..base_bbox_coder import BaseBBoxCoder
from mmdet.core.bbox.transforms_obb import regular_theta, regular_obb
from mmdet.core.bbox.builder import BBOX_CODERS
import pdb

pi = 3.141592


@BBOX_CODERS.register_module()
class HBB2OBBDeltaXYWHTCoder(BaseBBoxCoder):

    def __init__(self,
                 theta_norm=True,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.)):
        super(BaseBBoxCoder, self).__init__()
        self.theta_norm = theta_norm
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes):
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == 4
        assert gt_bboxes.size(-1) == 5
        encoded_bboxes = obb2delta(bboxes, gt_bboxes, self.theta_norm, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16/1000):
        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta2obb(bboxes, pred_bboxes, self.theta_norm, 
                                   self.means, self.stds, wh_ratio_clip)

        return decoded_bboxes


def obb2delta(proposals, gt, theta_norm=True, means=(0., 0., 0., 0., 0.), stds=(1., 1., 1., 1., 1.)):
    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]
    gx, gy, gw, gh, gtheta = gt.unbind(dim=-1)

    dtheta1 = regular_theta(gtheta)
    dtheta2 = regular_theta(gtheta + pi/2)
    abs_dtheta1 = torch.abs(dtheta1)
    abs_dtheta2 = torch.abs(dtheta2)

    gw_regular = torch.where(abs_dtheta1 < abs_dtheta2, gw, gh)
    gh_regular = torch.where(abs_dtheta1 < abs_dtheta2, gh, gw)
    dtheta = torch.where(abs_dtheta1 < abs_dtheta2, dtheta1, dtheta2)
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw_regular / pw)
    dh = torch.log(gh_regular / ph)

    if theta_norm:
        dtheta /= 2 * pi
    deltas = torch.stack([dx, dy, dw, dh, dtheta], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    return deltas


def delta2obb(proposals,
              deltas,
              theta_norm=True,
              means=(0., 0., 0., 0., 0.),
              stds=(1., 1., 1., 1., 1.),
              wh_ratio_clip=16/1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    dtheta = denorm_deltas[:, 4::5]
    if theta_norm:
        dtheta *= 2 * pi
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)

    px = ((proposals[:, 0] + proposals[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((proposals[:, 1] + proposals[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    pw = (proposals[:, 2] - proposals[:, 0]).unsqueeze(1).expand_as(dw)
    ph = (proposals[:, 3] - proposals[:, 1]).unsqueeze(1).expand_as(dh)

    gx = px + pw * dx
    gy = py + ph * dy
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gtheta = dtheta

    bboxes = torch.stack([gx, gy, gw, gh, gtheta], dim=-1)
    bboxes = regular_obb(bboxes)
    return bboxes.view_as(deltas)
