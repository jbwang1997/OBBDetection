import numpy as np
import torch

from . import box_iou_rotated_ext
from ..convex import convex_sort


def obb_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, device_id=None):
    assert mode in ['iou', 'iof']
    assert type(bboxes1) is type(bboxes2)
    if is_aligned:
        assert bboxes1.shape[0] == bboxes2.shape[0]

    if isinstance(bboxes1, torch.Tensor):
        is_numpy = False
        bboxes1_th = bboxes1
        bboxes2_th = bboxes2
    elif isinstance(bboxes1, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else f'cuda:{device_id}'
        bboxes1_th = torch.from_numpy(bboxes1).float().to(device)
        bboxes2_th = torch.from_numpy(bboxes2).float().to(device)
    else:
        raise TypeError('bboxes must be either a Tensor or numpy array, '
                        f'but got {type(bboxes1)}')

    if bboxes1_th.numel() == 0 or bboxes2_th.numel() == 0:
        rows, cols = bboxes1_th.size(0), bboxes2_th.size(0)
        outputs = bboxes1_th.new_zeros(rows, 1) if is_aligned else \
                bboxes1_th.new_zeros(rows, cols)
    elif is_aligned:
        outputs = aligned_obb_overlaps(
            bboxes1_th, bboxes2_th, mode)
    else:
        outputs = box_iou_rotated_ext.overlaps(
            bboxes1_th, bboxes2_th, mode == 'iou')

        # same bug will happen when bbox size is to small
        too_small1 = bboxes1_th[:, [2, 3]].min(1)[0] < 0.001
        too_small2 = bboxes2_th[:, [2, 3]].min(1)[0] < 0.001
        if too_small1.any() or too_small2.any():
            inds1 = torch.nonzero(too_small1, as_tuple=False)
            inds2 = torch.nonzero(too_small2, as_tuple=False)
            outputs[inds1, :] = 0.
            outputs[:, inds2] = 0.

    if is_numpy:
        outputs = outputs.cpu().numpy()
    return outputs


def aligned_obb_overlaps(bboxes1, bboxes2, mode='iou'):
    areas1 = bboxes1[:, 2] * bboxes1[:, 3]
    areas2 = bboxes2[:, 2] * bboxes2[:, 3]
    bboxes1, bboxes2 = obb2poly(bboxes1), obb2poly(bboxes2)

    num_objs = bboxes1.size(0)
    bboxes1_pts = bboxes1.view(num_objs, -1, 2)
    bboxes2_pts = bboxes2.view(num_objs, -1, 2)
    inter_pts, inter_masks = poly_intersection(
        bboxes1_pts, bboxes2_pts, areas1, areas2)
    overlap = convex_areas(inter_pts, inter_masks)

    if mode == 'iou':
        outputs = overlap / (areas1 + areas2 - overlap)
    else:
        outputs = overlap / areas1
    return outputs[..., None]


def obb2poly(obboxes):
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)

    vector1 = torch.cat(
        [w/2 * Cos, -w/2 * Sin], dim=-1)
    vector2 = torch.cat(
        [-h/2 * Sin, -h/2 * Cos], dim=-1)

    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    return torch.cat(
        [point1, point2, point3, point4], dim=-1)


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
