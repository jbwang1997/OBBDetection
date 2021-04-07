import numpy as np
import torch

from .form import hbb2poly, poly2hbb
from .misc import regular_theta
import pdb

pi = 3.141592


def hbb_flip(bboxes, img_shape, direction='horizontal'):
    assert bboxes.shape[-1] % 4 == 0
    assert direction in ['horizontal', 'vertical']
    flipped = bboxes.clone()
    if direction == 'horizontal':
        flipped[..., 0::4] = img_shape[1] - bboxes[..., 2::4]
        flipped[..., 2::4] = img_shape[1] - bboxes[..., 0::4]
    else:
        flipped[..., 1::4] = img_shape[0] - bboxes[..., 3::4]
        flipped[..., 3::4] = img_shape[0] - bboxes[..., 1::4]
    return flipped


def obb_flip(bboxes, img_shape, direction='horizontal'):
    assert bboxes.shape[-1] % 5 == 0
    assert direction in ['horizontal', 'vertical']
    flipped = bboxes.clone()
    if direction == 'horizontal':
        flipped[..., 0::5] = img_shape[1] - bboxes[..., 0::5]
    else:
        flipped[..., 1::5] = img_shape[0] - bboxes[..., 1::5]

    no_v = bboxes[..., 4::5] != -pi/2
    flipped[..., 4::5][no_v] = -bboxes[..., 4::5][no_v]
    return flipped


def poly_flip(bboxes, img_shape, direction='horizontal'):
    assert bboxes.shape[-1] % 8 == 0
    assert direction in ['horizontal', 'vertical']
    flipped = bboxes.clone()
    if direction == 'horizontal':
        flipped[..., 0::2] = img_shape[1] - bboxes[..., 0::2]
    else:
        flipped[..., 1::2] = img_shape[0] - bboxes[..., 1::2]
    return flipped


def pts_warp(pts, M):
    if pts.numel() == 0:
        return pts

    assert pts.shape[-1] % 2 == 0
    pts = pts.view(*pts.shape[:-1], -1, 2)
    aug = pts.new_ones((*pts.shape[:-1], 1))
    aug_pts = torch.cat([pts, aug], dim=-1)

    warped_pts = torch.matmul(aug_pts, M.transpose(0, 1))
    if warped_pts.size(-1) == 3:
        warped_pts = (warped_pts / warped_pts[..., -1:])[..., :-1]
    warped_pts = warped_pts.flatten(start_dim=-2, end_dim=-1)
    return warped_pts


def hbb_warp(bboxes, M):
    assert bboxes.shape[-1] % 4 == 0
    bboxes = bboxes.view(*bboxes.shape[:-1], -1, 4)
    polys = hbb2poly(bboxes)
    warped_polys = pts_warp(polys, M)

    bboxes = poly2hbb(warped_polys)
    new_bboxes = bboxes.flatten(start_dim=-2, end_dim=-1)
    return new_bboxes


def obb_warp(bboxes, angle, M):
    assert bboxes.shape[-1] % 5 == 0
    bboxes = bboxes.view(*bboxes.shape[:-1], -1, 5)

    center, size, theta = torch.split(bboxes, [2, 2, 1], dim=-1)
    warped_center = pts_warp(center, M)
    warped_theta = theta + angle/180*pi
    warped_theta = regular_theta(warped_theta)
    warped_obb = torch.cat([warped_center, size, warped_theta], dim=-1)

    return warped_obb.flatten(start_dim=-2, end_dim=-1)


def poly_warp(bboxes, M):
    assert bboxes.shape[-1] % 8 == 0
    return pts_warp(bboxes, M)


def hbb_mapping(bboxes,
                img_shape,
                scale_factor,
                h_flip,
                v_flip,
                rotate_after_flip,
                angle,
                matrix):
    assert angle % 90 == 0
    if isinstance(matrix, np.ndarray):
        matrix = bboxes.new_tensor(matrix)
    new_bboxes = bboxes * bboxes.new_tensor(scale_factor)
    action_order = ['flip', 'rotate'] if rotate_after_flip else \
            ['rotate', 'flip']
    for action in action_order:
        if action == 'flip':
            if h_flip:
                new_bboxes = hbb_flip(new_bboxes, img_shape, 'horizontal')
            if v_flip:
                new_bboxes = hbb_flip(new_bboxes, img_shape, 'vertical')
        if action == 'rotate':
            if angle != 0:
                new_bboxes = hbb_warp(new_bboxes, matrix)
    return new_bboxes


def hbb_mapping_back(bboxes,
                     img_shape,
                     scale_factor,
                     h_flip,
                     v_flip,
                     rotate_after_flip,
                     angle,
                     matrix):
    assert angle % 90 == 0
    if isinstance(matrix, np.ndarray):
        matrix = bboxes.new_tensor(matrix)
    action_order = ['rotate', 'flip'] if rotate_after_flip else \
            ['flip', 'rotate']
    for action in action_order:
        if action == 'flip':
            if h_flip:
                bboxes = hbb_flip(bboxes, img_shape, 'horizontal')
            if v_flip:
                bboxes = hbb_flip(bboxes, img_shape, 'vertical')
        if action == 'rotate':
            if angle != 0:
                if matrix.size(0) == 2:
                    _matrix = bboxes.new_tensor(
                        [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                    _matrix[:2] = matrix
                else:
                    _matrix = matrix
                _matrix  = torch.inverse(_matrix)
                bboxes = hbb_warp(bboxes, _matrix)
    new_bboxes = bboxes.view(-1, 4)
    new_bboxes = new_bboxes / new_bboxes.new_tensor(scale_factor)
    return new_bboxes.view(bboxes.shape)


def obb_mapping(bboxes,
                img_shape,
                scale_factor,
                h_flip,
                v_flip,
                rotate_after_flip,
                angle,
                matrix):
    if isinstance(matrix, np.ndarray):
        matrix = bboxes.new_tensor(matrix)
    new_bboxes = bboxes.clone()
    new_bboxes[:, :4] = bboxes[:, :4] * bboxes.new_tensor(scale_factor)
    action_order = ['flip', 'rotate'] if rotate_after_flip else \
            ['rotate', 'flip']
    for action in action_order:
        if action == 'flip':
            if h_flip:
                new_bboxes = obb_flip(new_bboxes, img_shape, 'horizontal')
            if v_flip:
                new_bboxes = obb_flip(new_bboxes, img_shape, 'vertical')
        if action == 'rotate':
            if angle != 0:
                new_bboxes = obb_warp(new_bboxes, angle, matrix)
    return new_bboxes


def obb_mapping_back(bboxes,
                     img_shape,
                     scale_factor,
                     h_flip,
                     v_flip,
                     rotate_after_flip,
                     angle,
                     matrix):
    if isinstance(matrix, np.ndarray):
        matrix = bboxes.new_tensor(matrix)
    action_order = ['rotate', 'flip'] if rotate_after_flip else \
            ['flip', 'rotate']
    for action in action_order:
        if action == 'flip':
            if h_flip:
                bboxes = obb_flip(bboxes, img_shape, 'horizontal')
            if v_flip:
                bboxes = obb_flip(bboxes, img_shape, 'vertical')
        if action == 'rotate':
            if angle != 0:
                _angle = -angle
                if matrix.size(0) == 2:
                    _matrix = bboxes.new_tensor(
                        [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                    _matrix[:2] = matrix
                else:
                    _matrix = matrix
                _matrix  = torch.inverse(_matrix)
                bboxes = obb_warp(bboxes, _angle, _matrix)
    new_bboxes = bboxes.view(-1, 5)
    new_bboxes[:, :4] = new_bboxes[:, :4] / new_bboxes.new_tensor(scale_factor)
    return new_bboxes.view(bboxes.shape)


def poly_mapping(bboxes,
                 img_shape,
                 scale_factor,
                 h_flip,
                 v_flip,
                 rotate_after_flip,
                 angle,
                 matrix):
    if isinstance(matrix, np.ndarray):
        matrix = bboxes.new_tensor(matrix)
    new_bboxes = bboxes * bboxes.new_tensor(scale_factor).repeat(2)
    action_order = ['flip', 'rotate'] if rotate_after_flip else \
            ['rotate', 'flip']
    for action in action_order:
        if action == 'flip':
            if h_flip:
                new_bboxes = poly_flip(new_bboxes, img_shape, 'horizontal')
            if v_flip:
                new_bboxes = poly_flip(new_bboxes, img_shape, 'vertical')
        if action == 'rotate':
            if angle != 0:
                new_bboxes = poly_warp(new_bboxes, matrix)
    return new_bboxes


def poly_mapping_back(bboxes,
                      img_shape,
                      scale_factor,
                      h_flip,
                      v_flip,
                      rotate_after_flip,
                      angle,
                      matrix):
    if isinstance(matrix, np.ndarray):
        matrix = bboxes.new_tensor(matrix)
    action_order = ['rotate', 'flip'] if rotate_after_flip else \
            ['flip', 'rotate']
    for action in action_order:
        if action == 'flip':
            if h_flip:
                bboxes = poly_flip(bboxes, img_shape, 'horizontal')
            if v_flip:
                bboxes = poly_flip(bboxes, img_shape, 'vertical')
        if action == 'rotate':
            if angle != 0:
                if matrix.size(0) == 2:
                    _matrix = bboxes.new_tensor(
                        [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                    _matrix[:2] = matrix
                else:
                    _matrix = matrix
                _matrix  = torch.inverse(_matrix)
                bboxes = poly_warp(bboxes, _matrix)
    new_bboxes = bboxes.view(-1, 8)
    new_bboxes = new_bboxes / new_bboxes.new_tensor(scale_factor).repeat(2)
    return new_bboxes.view(bboxes.shape)


def arb_mapping(bboxes,
                img_shape,
                scale_factor,
                h_flip,
                v_flip,
                rotate_after_flip,
                angle,
                matrix,
                bbox_type='hbb'):
    if bbox_type == 'hbb':
        mapping_func = hbb_mapping
    elif bbox_type == 'obb':
        mapping_func = obb_mapping
    else:
        mapping_func = poly_mapping

    return mapping_func(bboxes, img_shape, scale_factor,
                        h_flip, v_flip, rotate_after_flip,
                        angle, matrix)


def arb_mapping_back(bboxes,
                     img_shape,
                     scale_factor,
                     h_flip,
                     v_flip,
                     rotate_after_flip,
                     angle,
                     matrix,
                     bbox_type='hbb'):
    if bbox_type == 'hbb':
        mapping_back_func = hbb_mapping_back
    elif bbox_type == 'obb':
        mapping_back_func = obb_mapping_back
    else:
        mapping_back_func = poly_mapping_back

    return mapping_back_func(bboxes, img_shape, scale_factor,
                             h_flip, v_flip, rotate_after_flip,
                             angle, matrix)
