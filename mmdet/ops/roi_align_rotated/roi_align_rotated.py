import numpy as np
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from . import roi_align_rotated_ext


class RoIAlignRotatedFunction(Function):

    @staticmethod
    def forward(ctx,
                features,
                rois,
                out_size,
                spatial_scale,
                sample_num=0,
                aligned=True):
        out_h, out_w = _pair(out_size)
        assert isinstance(out_h, int) and isinstance(out_w, int)
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sample_num
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()
        ctx.aligned = aligned

        output = roi_align_rotated_ext.forward(
            features, rois, spatial_scale, out_h, out_w, sample_num, aligned)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        feature_size = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        sample_num = ctx.sample_num
        rois = ctx.saved_tensors[0]
        aligned = ctx.aligned
        assert feature_size is not None

        batch_size, num_channels, data_height, data_width = feature_size
        out_w = grad_output.size(3)
        out_h = grad_output.size(2)

        grad_input = grad_rois = None
        grad_input = roi_align_rotated_ext.backward(
            grad_output, rois, spatial_scale, out_h, out_w,
            batch_size, num_channels, data_height, data_width,
            sample_num, aligned)

        return grad_input, grad_rois, None, None, None, None


roi_align_rotated = RoIAlignRotatedFunction.apply


class RoIAlignRotated(nn.Module):

    def __init__(self,
                 out_size,
                 spatial_scale,
                 sample_num=0,
                 aligned=True):
        super(RoIAlignRotated, self).__init__()
        self.out_size = _pair(out_size)
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.aligned = aligned

    def forward(self, features, rois):
        assert rois.dim() == 2 and rois.size(1) == 6
        return roi_align_rotated(features, rois, self.out_size,
                                 self.spatial_scale, self.sample_num,
                                 self.aligned)

    def __repr__(self):
        indent_str = '\n    '
        format_str = self.__class__.__name__
        format_str += f'({indent_str}out_size={self.out_size},'
        format_str += f'{indent_str}spatial_scale={self.spatial_scale},'
        format_str += f'{indent_str}sample_num={self.sample_num},'
        format_str += f'{indent_str}aligned={self.aligned},'
        return format_str
