from copy import deepcopy

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init
from mmcv.ops import DeformConv2dPack, DeformConv2d

from mmdet.core import get_bbox_dim
from mmdet.models.builder import HEADS, build_head
from ..base_dense_head import BaseDenseHead


class AlignConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deform_groups=1):
        super(AlignConv, self).__init__()
        self.kernel_size = kernel_size
        self.deform_conv = DeformConv2d(in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        padding=(kernel_size - 1) // 2,
                                        deform_groups=deform_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.deform_conv, std=0.01)

    @torch.no_grad()
    def get_offset(self, anchors, featmap_size, stride):
        dtype, device = anchors.dtype, anchors.device
        feat_h, feat_w = featmap_size

        # Check if stride is tuple
        if isinstance(stride, tuple):
            assert stride[0] == stride[1]
            stride = stride[0]

        pad = (self.kernel_size - 1) // 2
        idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(idx, idx)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        # get sampling locations of default conv
        xc = torch.arange(0, feat_w, device=device, dtype=dtype)
        yc = torch.arange(0, feat_h, device=device, dtype=dtype)
        yc, xc = torch.meshgrid(yc, xc)
        xc = xc.reshape(-1)
        yc = yc.reshape(-1)
        x_conv = xc[:, None] + xx
        y_conv = yc[:, None] + yy

        # get sampling locations of anchors
        x_ctr, y_ctr, w, h, a = torch.unbind(anchors, dim=1)
        x_ctr, y_ctr, w, h = x_ctr / stride, y_ctr / stride, w / stride, h / stride
        cos, sin = torch.cos(-a), torch.sin(-a)
        dw, dh = w / self.kernel_size, h / self.kernel_size
        x, y = dw[:, None] * xx, dh[:, None] * yy
        xr = cos[:, None] * x - sin[:, None] * y
        yr = sin[:, None] * x + cos[:, None] * y
        x_anchor, y_anchor = xr + x_ctr[:, None], yr + y_ctr[:, None]
        # get offset filed
        offset_x = x_anchor - x_conv
        offset_y = y_anchor - y_conv
        # x, y in anchors is opposite in image coordinates,
        # so we stack them with y, x other than x, y
        offset = torch.stack([offset_y, offset_x], dim=-1)
        # NA,ks*ks*2
        offset = offset.reshape(anchors.size(
            0), -1).permute(1, 0).reshape(-1, feat_h, feat_w)
        return offset

    def forward(self, x, anchors, stride):
        num_imgs, H, W = anchors.shape[:3]
        offset_list = [
            self.get_offset(anchors[i].reshape(-1, 5), (H, W), stride)
            for i in range(num_imgs)
        ]
        offset_tensor = torch.stack(offset_list, dim=0)
        x = self.relu(self.deform_conv(x, offset_tensor))
        return x


@HEADS.register_module()
class S2AHead(BaseDenseHead):

    def __init__(self,
                 heads,
                 feat_channels=256,
                 align_type='AlignConv',
                 train_cfg=None,
                 test_cfg=None):
        super(S2AHead, self).__init__()

        # Skip cls branch to speedup inference speed
        if isinstance(test_cfg['skip_cls'], list):
            skip_cls = test_cfg['skip_cls']
            test_cfg.pop('skip_cls')
            assert len(skip_cls) == len(heads)
            test_cfg = [deepcopy(test_cfg) for _ in range(len(heads))]
            for skip, cfg in zip(skip_cls, test_cfg):
                cfg['skip_cls'] = skip
        else:
            test_cfg = [test_cfg for _ in heads]

        self.heads = nn.ModuleList()
        for i, head in enumerate(heads):
            if train_cfg is not None:
                head.update(train_cfg=train_cfg[i])

            head.update(test_cfg=test_cfg[i])
            head_module = build_head(head)
            if i == 0:
                self.anchor_generator = head_module.anchor_generator
                self.num_anchors = head_module.num_anchors
            else:
                head_module.anchor_generator = self.anchor_generator
                head_module.num_anchors = self.num_anchors
            self.heads.append(head_module)

        self.num_stages = len(self.heads)
        self.num_classes = self.heads[-1].num_classes

        assert self.num_stages >= 2

        if isinstance(align_type, str):
            self.align_type = [align_type for _ in range(len(self.heads) - 1)]
        else:
            assert len(align_type) == len(self.heads) - 1
            self.align_type = align_type

        self.feat_channels = feat_channels
        self.bbox_type = 'obb'
        self.reg_dim = get_bbox_dim(self.bbox_type)
        self._init_layers()

    def _init_layers(self):
        """Initialize Align layers of the S2AHead."""
        self.align_convs = nn.ModuleList()
        for align_type in self.align_type:

            assert align_type in ['Conv', 'DCN', 'AlignConv']
            if align_type == 'Conv':
                self.align_convs.append(ConvModule(self.feat_channels,
                                                   self.feat_channels,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1))
            elif align_type == 'DCN':
                self.align_convs.append(DeformConv2dPack(self.feat_channels,
                                                         self.feat_channels,
                                                         kernel_size=3,
                                                         stride=1,
                                                         padding=1,
                                                         deform_groups=1))
            elif align_type == 'AlignConv':
                self.align_convs.append(AlignConv(self.feat_channels,
                                                  self.feat_channels,
                                                  kernel_size=3,
                                                  deform_groups=1))

    def init_weights(self):
        """Initialize weights of the head."""
        for head in self.heads:
            head.init_weights()
        for align_conv in self.align_convs:
            align_conv.init_weights()

    def loss(self, **kwargs):
        raise Exception

    def bbox_decode(self,
                    bbox_preds,
                    anchors,
                    stage):
        """Decode bboxes from deltas"""
        bboxes_list = []
        for pred, anchor in zip(bbox_preds, anchors):
            pred = pred.detach()
            anchor = anchor.repeat(pred.size()[0], 1, 1).reshape(-1, 5)
            num_imgs, _, H, W = pred.shape

            bbox_delta = pred.permute(0, 2, 3, 1).reshape(-1, 5)
            bboxes = self.heads[stage].bbox_coder.decode(anchor, bbox_delta, wh_ratio_clip=1e-6)
            bboxes = bboxes.reshape(-1, H, W, 5)
            bboxes_list.append(bboxes)
        return bboxes_list

    def get_pred_anchors(self, outs, stage, prior_anchors=None):
        """Generate anchors from head's out"""
        if prior_anchors is None:
            featmap_size = [feat.size()[2:4] for feat in outs[1]]
            prior_anchors = self.heads[stage].anchor_generator.grid_anchors(
                featmap_size, device=outs[1][0].device)
        return self.bbox_decode(outs[1], prior_anchors, stage)

    def align_feature(self, stage, x, proposals):
        """Align Convolutional Feature By Proposals"""
        align_type = self.align_type[stage]
        align_conv = self.align_convs[stage]
        align_feats = []
        if align_type == 'AlignConv':
            for feat, proposal, stride in zip(x, proposals, self.anchor_generator.strides):
                p2 = proposal.clone()
                # p2[..., -1] = -p2[..., -1]
                align_feats.append(align_conv(feat, p2, stride))
        else:
            for feat in x:
                align_feats.append(align_conv(feat))

        return align_feats

    def _bbox_forward_train(self, stage, x, gt_obboxes, gt_labels, img_metas, gt_bboxes_ignore=None,
                            prior_anchors=None, with_anchor=False):
        """Run forward function and calculate loss for box head in training"""
        outs = self.heads[stage](x)

        loss_inputs = outs + (gt_obboxes, gt_labels, prior_anchors, img_metas)
        losses = self.heads[stage].loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        result_dict = dict()
        result_dict.update(losses=losses)
        if with_anchor:
            result_dict.update(prior_anchors=self.get_pred_anchors(outs, stage, prior_anchors))
        return result_dict

    def forward(self, feats):
        prior_anchors = None
        outs = None
        for i in range(self.num_stages):
            with_anchor = True if i != (self.num_stages - 1) else False
            outs = self.heads[i](feats)
            if with_anchor:
                prior_anchors = self.get_pred_anchors(outs, i, prior_anchors)
                feats = self.align_feature(i, feats, prior_anchors)
        return outs + (prior_anchors,)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_obboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        losses = dict()
        feat = x
        prior_anchors = None
        for i in range(self.num_stages):
            with_anchor = True if i != (self.num_stages - 1) else False
            result_dict = self._bbox_forward_train(i, feat, gt_obboxes,
                                                   gt_labels, img_metas,
                                                   gt_bboxes_ignore, prior_anchors,
                                                   with_anchor=with_anchor)
            # update loss
            for name, value in result_dict['losses'].items():
                losses[f's{i}.{name}'] = value

            # align feature
            if with_anchor:
                prior_anchors = result_dict['prior_anchors']
                feat = self.align_feature(i, feat, prior_anchors)

        return losses

    def get_bboxes(self, *args, **kwargs):
        return self.heads[-1].get_bboxes(*args, **kwargs)
