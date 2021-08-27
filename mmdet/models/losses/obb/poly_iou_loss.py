import torch
import torch.nn as nn

from mmdet.ops import convex_sort
from mmdet.core import bbox2type, get_bbox_areas
from mmdet.models.builder import LOSSES
from ..utils import weighted_loss


def shoelace(pts):
    roll_pts = torch.roll(pts, 1, dims=-2)
    xyxy = pts[..., 0] * roll_pts[..., 1] - \
           roll_pts[..., 0] * pts[..., 1]
    areas = 0.5 * torch.abs(xyxy.sum(dim=-1))
    return areas


def convex_areas(pts, masks):
    nbs, npts, _ = pts.size()
    index = convex_sort(pts, masks)
    index[index == -1] = npts
    index = index[..., None].repeat(1, 1, 2)

    ext_zeros = pts.new_zeros((nbs, 1, 2))
    ext_pts = torch.cat([pts, ext_zeros], dim=1)
    polys = torch.gather(ext_pts, 1, index)

    xyxy = polys[:, 0:-1, 0] * polys[:, 1:, 1] - \
           polys[:, 0:-1, 1] * polys[:, 1:, 0]
    areas = 0.5 * torch.abs(xyxy.sum(dim=-1))
    return areas


def poly_intersection(pts1, pts2, areas1=None, areas2=None, eps=1e-6):
    # Calculate the intersection points and the mask of whether points is inside the lines.
    # Reference:
    #    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    #    https://github.com/lilanxiao/Rotated_IoU/blob/master/box_intersection_2d.py
    lines1 = torch.cat([pts1, torch.roll(pts1, -1, dims=1)], dim=2)
    lines2 = torch.cat([pts2, torch.roll(pts2, -1, dims=1)], dim=2)
    lines1, lines2 = lines1.unsqueeze(2), lines2.unsqueeze(1)
    x1, y1, x2, y2 = lines1.unbind(dim=-1) # dim: N, 4, 1
    x3, y3, x4, y4 = lines2.unbind(dim=-1) # dim: N, 1, 4

    num = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    den_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    with torch.no_grad():
        den_u = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
        t, u = den_t / num, den_u / num
        mask_t = (t > 0) & (t < 1)
        mask_u = (u > 0) & (u < 1)
        mask_inter = torch.logical_and(mask_t, mask_u)

    t = den_t / (num + eps)
    x_inter = x1 + t * (x2 - x1)
    y_inter = y1 + t * (y2 - y1)
    pts_inter = torch.stack([x_inter, y_inter], dim=-1)

    B = pts1.size(0)
    pts_inter = pts_inter.view(B, -1, 2)
    mask_inter = mask_inter.view(B, -1)

    # Judge if one polygon's vertices are inside another polygon.
    # Use
    with torch.no_grad():
        areas1 = shoelace(pts1) if areas1 is None else areas1
        areas2 = shoelace(pts2) if areas2 is None else areas2

        triangle_areas1 = 0.5 * torch.abs(
            (x3 - x1) * (y4 - y1) - (y3 - y1) * (x4 - x1))
        sum_areas1 = triangle_areas1.sum(dim=-1)
        mask_inside1 = torch.abs(sum_areas1 - areas2[..., None]) < 1e-3 * areas2[..., None]

        triangle_areas2 = 0.5 * torch.abs(
            (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3))
        sum_areas2 = triangle_areas2.sum(dim=-2)
        mask_inside2 = torch.abs(sum_areas2 - areas1[..., None]) < 1e-3 * areas1[..., None]

    all_pts = torch.cat([pts_inter, pts1, pts2], dim=1)
    masks = torch.cat([mask_inter, mask_inside1, mask_inside2], dim=1)
    return all_pts, masks


def poly_enclose(pts1, pts2):
    all_pts = torch.cat([pts1, pts2], dim=1)
    mask1 = pts1.new_ones((pts1.size(0), pts1.size(1)))
    mask2 = pts2.new_ones((pts2.size(0), pts2.size(1)))
    masks = torch.cat([mask1, mask2], dim=1)
    return all_pts, masks


@weighted_loss
def poly_iou_loss(pred, target, linear=False, eps=1e-6):
    areas1, areas2 = get_bbox_areas(pred), get_bbox_areas(target)
    pred, target = bbox2type(pred, 'poly'), bbox2type(target, 'poly')

    pred_pts = pred.view(pred.size(0), -1, 2)
    target_pts = target.view(target.size(0), -1, 2)
    inter_pts, inter_masks = poly_intersection(
        pred_pts, target_pts, areas1, areas2, eps)
    overlap = convex_areas(inter_pts, inter_masks)

    ious = (overlap / (areas1 + areas2 - overlap + eps)).clamp(min=eps)
    if linear:
        loss = 1 - ious
    else:
        loss = -ious.log()
    return loss


@weighted_loss
def poly_giou_loss(pred, target, eps=1e-6):
    areas1, areas2 = get_bbox_areas(pred), get_bbox_areas(target)
    pred, target = bbox2type(pred, 'poly'), bbox2type(target, 'poly')

    pred_pts = pred.view(pred.size(0), -1, 2)
    target_pts = target.view(target.size(0), -1, 2)
    inter_pts, inter_masks = poly_intersection(
        pred_pts, target_pts, areas1, areas2, eps)
    overlap = convex_areas(inter_pts, inter_masks)

    union = areas1 + areas2 - overlap + eps
    ious = (overlap / union).clamp(min=eps)

    enclose_pts, enclose_masks = poly_enclose(pred_pts, target_pts)
    enclose_areas = convex_areas(enclose_pts, enclose_masks)

    gious = ious - (enclose_areas - union) / enclose_areas
    loss = 1 - gious
    return loss


@LOSSES.register_module()
class PolyIoULoss(nn.Module):

    def __init__(self,
                 linear=False,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0):
        super(PolyIoULoss, self).__init__()
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * poly_iou_loss(
            pred,
            target,
            weight,
            linear=self.linear,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module()
class PolyGIoULoss(nn.Module):

    def __init__(self,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0):
        super(PolyGIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * poly_giou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
