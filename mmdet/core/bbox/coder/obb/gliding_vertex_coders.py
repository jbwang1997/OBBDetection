import numpy as np
import torch

from ..base_bbox_coder import BaseBBoxCoder
from mmdet.core.bbox.transforms_obb import poly2hbb
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class GVFixCoder(BaseBBoxCoder):

    def encode(self, polys):
        assert polys.size(1) == 8
        max_x, max_x_idx = polys[:, ::2].max(1)
        min_x, min_x_idx = polys[:, ::2].min(1)
        max_y, max_y_idx = polys[:, 1::2].max(1)
        min_y, min_y_idx = polys[:, 1::2].min(1)
        hbboxes = torch.stack([min_x, min_y, max_x, max_y], dim=1)

        polys = polys.view(-1, 4, 2)
        num_polys = polys.size(0)
        polys_ordered = torch.zeros_like(polys)
        polys_ordered[:, 0] = polys[range(num_polys), min_y_idx]
        polys_ordered[:, 1] = polys[range(num_polys), max_x_idx]
        polys_ordered[:, 2] = polys[range(num_polys), max_y_idx]
        polys_ordered[:, 3] = polys[range(num_polys), min_x_idx]

        t_x = polys_ordered[:, 0, 0]
        r_y = polys_ordered[:, 1, 1]
        d_x = polys_ordered[:, 2, 0]
        l_y = polys_ordered[:, 3, 1]

        dt = (t_x - hbboxes[:, 0]) / (hbboxes[:, 2] - hbboxes[:, 0])
        dr = (r_y - hbboxes[:, 1]) / (hbboxes[:, 3] - hbboxes[:, 1])
        dd = (hbboxes[:, 2] - d_x) / (hbboxes[:, 2] - hbboxes[:, 0])
        dl = (hbboxes[:, 3] - l_y) / (hbboxes[:, 3] - hbboxes[:, 1])

        h_mask = (polys_ordered[:, 0, 1] - polys_ordered[:, 1, 1] == 0) | \
                (polys_ordered[:, 1, 0] - polys_ordered[:, 2, 0] == 0)
        fix_deltas = torch.stack([dt, dr, dd, dl], dim=1)
        fix_deltas[h_mask, :] = 1
        return fix_deltas

    def decode(self, hbboxes, fix_deltas):
        x1 = hbboxes[:, 0::4]
        y1 = hbboxes[:, 1::4]
        x2 = hbboxes[:, 2::4]
        y2 = hbboxes[:, 3::4]
        w = hbboxes[:, 2::4] - hbboxes[:, 0::4]
        h = hbboxes[:, 3::4] - hbboxes[:, 1::4]

        pred_t_x = x1 + w * fix_deltas[:, 0::4]
        pred_r_y = y1 + h * fix_deltas[:, 1::4]
        pred_d_x = x2 - w * fix_deltas[:, 2::4]
        pred_l_y = y2 - h * fix_deltas[:, 3::4]

        polys = torch.stack([pred_t_x, y1,
                             x2, pred_r_y,
                             pred_d_x, y2,
                             x1, pred_l_y], dim=-1)
        polys = polys.flatten(1)
        return polys


@BBOX_CODERS.register_module()
class GVRatioCoder(BaseBBoxCoder):

    def encode(self, polys):
        assert polys.size(1) == 8
        hbboxes = poly2hbb(polys)
        h_areas = (hbboxes[:, 2] - hbboxes[:, 0]) * \
                (hbboxes[:, 3] - hbboxes[:, 1])

        polys = polys.view(polys.size(0), 4, 2)
        areas = polys.new_zeros(polys.size(0))
        for i in range(4):
            areas += 0.5 * (polys[:, i, 0] * polys[:, (i+1)%4, 1] -
                            polys[:, (i+1)%4, 0] * polys[:, i, 1])
        areas = torch.abs(areas)

        ratios = areas / h_areas
        return ratios[:, None]

    def decode(self, bboxes, bboxes_pred):
        raise NotImplementedError
