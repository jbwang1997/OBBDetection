import mmcv
import numpy as np
import torch

from mmdet.core.bbox.transforms_obb import regular_theta
from ..base_bbox_coder import BaseBBoxCoder
from ...builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class DeltaXYWHABBoxCoder(BaseBBoxCoder):
    """Delta XYWH BBox coder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh) and
    decodes delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.),
                 clip_border=True):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds
        self.clip_border = clip_border

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 5
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            bboxes (torch.Tensor): Basic boxes. Shape (B, N, 4) or (N, 4)
            pred_bboxes (Tensor): Encoded offsets with respect to each roi.
               Has shape (B, N, num_classes * 4) or (B, N, 4) or
               (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
               when rois is a grid of anchors.Offset encoding follows [1]_.
            max_shape (Sequence[int] or torch.Tensor or Sequence[
               Sequence[int]],optional): Maximum bounds for boxes, specifies
               (H, W, C) or (H, W). If bboxes shape is (B, N, 4), then
               the max_shape should be a Sequence[Sequence[int]]
               and the length of max_shape should also be B.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            torch.Tensor: Decoded boxes.
        """

        assert pred_bboxes.size(0) == bboxes.size(0)
        if pred_bboxes.ndim == 3:
            assert pred_bboxes.size(1) == bboxes.size(1)
        decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                                    max_shape, wh_ratio_clip, self.clip_border)

        return decoded_bboxes


@mmcv.jit(coderize=True)
def bbox2delta(proposals, gt, means=(0., 0., 0., 0., 0.), stds=(1., 1., 1., 1., 1.), angle_type='long_edge'):
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of :func:`delta2bbox`.

    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 4)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 4)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, 4), where columns represent dx, dy,
            dw, dh.
    """
    assert proposals.size() == gt.size()

    gt_widths = gt[..., 2]
    gt_heights = gt[..., 3]
    gt_angle = gt[..., 4]

    proposals_widths = proposals[..., 2]
    proposals_heights = proposals[..., 3]
    proposals_angles = proposals[..., 4]

    cosa = torch.cos(proposals_angles)
    sina = torch.sin(proposals_angles)
    coord = gt[..., 0:2] - proposals[..., 0:2]

    dx = (cosa * coord[..., 0] + sina * coord[..., 1]) / proposals_widths
    dy = (-sina * coord[..., 0] + cosa * coord[..., 1]) / proposals_heights
    dw = torch.log(gt_widths / proposals_widths)
    dh = torch.log(gt_heights / proposals_heights)
    da = gt_angle - proposals_angles
    # da = regular_theta(da, start=-np.pi/4) / np.pi
    da = regular_theta(da) / np.pi


    deltas = torch.stack((dx, dy, dw, dh, da), -1)
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


@mmcv.jit(coderize=True)
def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0.),
               stds=(1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               clip_border=True,
               angle_type='long_edge'):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4) or (B, N, 4)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (B, N, num_classes * 4) or (B, N, 4) or
            (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
            when rois is a grid of anchors.Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If rois shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.
        wh_ratio_clip (float): Maximum aspect ratio for boxes.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.

    Returns:
        Tensor: Boxes with shape (B, N, num_classes * 4) or (B, N, 4) or
           (N, num_classes * 4) or (N, 4), where 4 represent
           tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Example:
        >>> rois = torch.Tensor([[ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 5.,  5.,  5.,  5.]])
        >>> deltas = torch.Tensor([[  0.,   0.,   0.,   0.],
        >>>                        [  1.,   1.,   1.,   1.],
        >>>                        [  0.,   0.,   2.,  -1.],
        >>>                        [ 0.7, -1.9, -0.5,  0.3]])
        >>> delta2bbox(rois, deltas, max_shape=(32, 32, 3))
        tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                [0.1409, 0.1409, 2.8591, 2.8591],
                [0.0000, 0.3161, 4.1945, 0.6839],
                [5.0000, 5.0000, 5.0000, 5.0000]])
    """
    means = deltas.new_tensor(means).view(1,
                                          -1).repeat(1,
                                                     deltas.size(-1) // 5)
    stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(-1) // 5)
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[..., 0::5]
    dy = denorm_deltas[..., 1::5]
    dw = denorm_deltas[..., 2::5]
    dh = denorm_deltas[..., 3::5]
    dangle = denorm_deltas[..., 4::5]

    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    roi_x = (rois[..., 0]).unsqueeze(-1).expand_as(dx)
    roi_y = (rois[..., 1]).unsqueeze(-1).expand_as(dy)
    roi_w = (rois[..., 2]).unsqueeze(-1).expand_as(dw)
    roi_h = (rois[..., 3]).unsqueeze(-1).expand_as(dh)
    roi_angle = (rois[..., 4]).unsqueeze(-1).expand_as(dangle)
    gx = dx * roi_w * torch.cos(roi_angle) \
         - dy * roi_h * torch.sin(roi_angle) + roi_x
    gy = dx * roi_w * torch.sin(roi_angle) \
         + dy * roi_h * torch.cos(roi_angle) + roi_y
    gw = roi_w * dw.exp()
    gh = roi_h * dh.exp()


    ga = np.pi * dangle + roi_angle
    # ga = regular_theta(ga, start=-np.pi/4)
    ga = regular_theta(ga)


    if clip_border and max_shape is not None:
        # TODO: 需要补充
        pass

    bboxes = torch.stack([gx, gy, gw, gh, ga], dim=-1).view_as(deltas)
    return bboxes
