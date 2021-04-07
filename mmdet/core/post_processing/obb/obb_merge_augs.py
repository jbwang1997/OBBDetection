import numpy as np
import torch

from mmdet.ops import nms, obb_nms, poly_nms
from mmdet.core import regular_theta, choice_by_type
from mmdet.core import hbb_mapping_back, obb_mapping_back, poly_mapping_back

pi = 3.141592


def merge_rotate_aug_proposals(aug_proposals, img_metas, rpn_test_cfg):
    recovered_proposals = []
    mapping_back_func = choice_by_type(
        hbb_mapping_back, obb_mapping_back, poly_mapping_back,
        aug_proposals[0], with_score=True)
    nms_ops = choice_by_type(nms, obb_nms, poly_nms,
                             aug_proposals[0], with_score=True)

    for proposals, img_info in zip(aug_proposals, img_metas):
        img_shape = img_info['img_shape']
        scale_factor = img_info['scale_factor']
        angle = img_info.get('angle', 0)
        matrix = img_info.get('matrix', np.eye(3))
        rotate_after_flip = img_info.get('rotate_after_flip', True)
        if 'flip' in img_info:
            direction = img_info['flip_direction']
            h_flip = img_info['flip'] if direction == 'horizontal' else False
            v_flip = img_info['flip'] if direction == 'vertical' else False
        else:
            h_flip, v_flip = img_info['h_flip'], img_info['v_flip']
        _proposals = proposals.clone()
        _proposals[:, :-1] = mapping_back_func(_proposals[:, :-1], img_shape,
                                               scale_factor, h_flip, v_flip,
                                               rotate_after_flip, angle,
                                               matrix)
        recovered_proposals.append(_proposals)
    aug_proposals = torch.cat(recovered_proposals, dim=0)
    merged_proposals, _ = nms_ops(aug_proposals, rpn_test_cfg.nms_thr)
    scores = merged_proposals[:, -1]
    _, order = scores.sort(0, descending=True)
    num = min(rpn_test_cfg.max_num, merged_proposals.shape[0])
    order = order[:num]
    merged_proposals = merged_proposals[order, :]
    return merged_proposals


def merge_rotate_aug_hbb(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg,
                         merge_type='avg'):
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']

        angle = img_info[0].get('angle', 0)
        matrix = img_info[0].get('matrix', np.eye(3))
        rotate_after_flip = img_info[0].get('rotate_after_flip', True)
        if 'flip' in img_info[0]:
            direction = img_info[0]['flip_direction']
            h_flip = img_info[0]['flip'] if direction == 'horizontal' else False
            v_flip = img_info[0]['flip'] if direction == 'vertical' else False
        else:
            h_flip, v_flip = img_info[0]['h_flip'], img_info[0]['v_flip']
        bboxes = hbb_mapping_back(bboxes, img_shape, scale_factor, h_flip,
                                  v_flip, rotate_after_flip, angle, matrix)
        recovered_bboxes.append(bboxes)

    if merge_type == 'cat':
        bboxes = torch.cat(recovered_bboxes, dim=0)
    else:
        bboxes = torch.stack(recovered_bboxes).mean(dim=0)

    if aug_scores is None:
        return bboxes
    else:
        if merge_type == 'cat':
            scores = torch.cat(aug_scores, dim=0)
        else:
            scores = torch.stack(aug_scores).mean(dim=0)
        return bboxes, scores


def merge_rotate_aug_obb(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg,
                         merge_type='avg'):
    assert merge_type in ['cat', 'avg']
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']

        angle = img_info[0].get('angle', 0)
        matrix = img_info[0].get('matrix', np.eye(3))
        rotate_after_flip = img_info[0].get('rotate_after_flip', True)
        if 'flip' in img_info[0]:
            direction = img_info[0]['flip_direction']
            h_flip = img_info[0]['flip'] if direction == 'horizontal' else False
            v_flip = img_info[0]['flip'] if direction == 'vertical' else False
        else:
            h_flip, v_flip = img_info[0]['h_flip'], img_info[0]['v_flip']
        bboxes = obb_mapping_back(bboxes, img_shape, scale_factor, h_flip,
                                  v_flip, rotate_after_flip, angle, matrix)
        recovered_bboxes.append(bboxes)

    if merge_type == 'cat':
        bboxes = torch.cat(recovered_bboxes, dim=0)
    else:
        all_bboxes = torch.stack(recovered_bboxes)
        base_bboxes = recovered_bboxes[0].unsqueeze(0).expand_as(all_bboxes)
        deltas = _sub_obb(base_bboxes, all_bboxes).mean(0)
        bboxes = recovered_bboxes[0] + deltas

    if aug_scores is None:
        return bboxes
    else:
        if merge_type == 'cat':
            scores = torch.cat(aug_scores, dim=0)
        else:
            scores = torch.stack(aug_scores).mean(dim=0)
        return bboxes, scores


def _sub_obb(obb1, obb2):
    obb1 = obb1.view(*obb1.shape[:-1], -1, 5)
    obb2 = obb2.view(*obb2.shape[:-1], -1, 5)
    x1, y1, w1, h1, theta1 = obb1.unbind(dim=-1)
    x2, y2, w2, h2, theta2 = obb2.unbind(dim=-1)

    dtheta_h = regular_theta(theta2 - theta1)
    dtheta_v = regular_theta(theta2 - theta1 + pi/2)
    abs_dtheta_h = torch.abs(dtheta_h)
    abs_dtheta_v = torch.abs(dtheta_v)

    w2_regular = torch.where(abs_dtheta_h < abs_dtheta_v, w2, h2)
    h2_regular = torch.where(abs_dtheta_h < abs_dtheta_v, h2, w2)
    dtheta = torch.where(abs_dtheta_h < abs_dtheta_v, dtheta_h, dtheta_v)
    dx = x2 - x1
    dy = y2 - y1
    dw = w2_regular - w1
    dh = h2_regular - h1

    delta = torch.stack([dx, dy, dw, dh, dtheta], dim=-1)
    return delta.flatten(start_dim=-2, end_dim=-1)


def merge_rotate_aug_poly(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg,
                          merge_type='avg'):
    assert merge_type in ['cat', 'avg']
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']

        angle = img_info[0].get('angle', 0)
        matrix = img_info[0].get('matrix', np.eye(3))
        rotate_after_flip = img_info[0].get('rotate_after_flip', True)
        if 'flip' in img_info[0]:
            direction = img_info[0]['flip_direction']
            h_flip = img_info[0]['flip'] if direction == 'horizontal' else False
            v_flip = img_info[0]['flip'] if direction == 'vertical' else False
        else:
            h_flip, v_flip = img_info[0]['h_flip'], img_info[0]['v_flip']
        bboxes = poly_mapping_back(bboxes, img_shape, scale_factor, h_flip,
                                   v_flip, rotate_after_flip, angle, matrix)
        recovered_bboxes.append(bboxes)

    if merge_type == 'cat':
        bboxes = torch.cat(recovered_bboxes, dim=0)
    else:
        all_bboxes = torch.stack(recovered_bboxes)
        base_bboxes = recovered_bboxes[0].unsqueeze(0).expand_as(all_bboxes)
        deltas = _sub_poly(base_bboxes, all_bboxes).mean(0)
        bboxes = recovered_bboxes[0] + deltas

    if aug_scores is None:
        return bboxes
    else:
        if merge_type == 'cat':
            scores = torch.cat(aug_scores, dim=0)
        else:
            scores = torch.stack(aug_scores).mean(dim=0)
        return bboxes, scores


def _sub_poly(poly1, poly2):
    poly1 = poly1.view(*poly1.shape[:-1], -1, 8)
    poly2 = poly2.view(*poly2.shape[:-1], -1, 8)
    _poly2 = poly2.view(*poly2.shape[:-1], 4, 2)
    _poly2 = torch.flip(_poly2, [-2]).flatten(-2)
    all_polys = []
    for i in range(4):
        all_polys.append(
            torch.cat([poly2[..., 2*i:], poly2[..., :2*i]], dim=-1)
        )
        all_polys.append(
            torch.cat([_poly2[..., 2*i:], _poly2[..., :2*i]], dim=-1)
        )

    all_polys = torch.stack(all_polys, dim=-1)
    poly1 = poly1.unsqueeze(-1).expand_as(all_polys)
    all_deltas = all_polys - poly1

    delta_dist = torch.square(all_deltas).sum(dim=-2)
    _, min_dist_inds = torch.min(delta_dist, dim=-1)
    min_dist_inds = min_dist_inds[..., None, None].expand(
        (*min_dist_inds.shape, 8, 1))

    deltas = torch.gather(all_deltas, -1, min_dist_inds)
    deltas = deltas.flatten(-3)
    return deltas


def merge_rotate_aug_arb(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg,
                         merge_type='avg', bbox_type='hbb'):
    merge_func = choice_by_type(merge_rotate_aug_hbb,
                                merge_rotate_aug_obb,
                                merge_rotate_aug_poly,
                                bbox_type)
    return merge_func(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg, merge_type)
