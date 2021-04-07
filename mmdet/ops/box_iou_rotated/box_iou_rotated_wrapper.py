import numpy as np
import torch

from . import box_iou_rotated_ext


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
    else:
        outputs = box_iou_rotated_ext.overlaps(
            bboxes1_th,
            bboxes2_th,
            mode == 'iou')

        # same bug will happen when bbox size is to small
        too_small1 = bboxes1_th[:, [2, 3]].min(1)[0] < 0.001
        too_small2 = bboxes2_th[:, [2, 3]].min(1)[0] < 0.001
        if too_small1.any() or too_small2.any():
            inds1 = torch.nonzero(too_small1, as_tuple=False)
            inds2 = torch.nonzero(too_small2, as_tuple=False)
            outputs[inds1, :] = 0.
            outputs[:, inds2] = 0.

    if is_aligned:
        eye_index = torch.arange(bboxes1.shape[0])[..., None]
        outputs = torch.gather(outputs, dim=1, index=eye_index)
    if is_numpy:
        outputs = outputs.cpu().numpy()
    return outputs
