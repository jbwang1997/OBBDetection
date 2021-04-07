import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmdet.core import (auto_fp16, build_bbox_coder, force_fp32, multi_apply,
                        multiclass_arb_nms, hbb2poly, bbox2type)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy


@HEADS.register_module()
class GVBBoxHead(nn.Module):

    def __init__(self,
                 with_avg_pool=False,
                 num_shared_fcs=2,
                 roi_feat_size=7,
                 in_channels=256,
                 fc_out_channels=1024,
                 num_classes=15,
                 reg_class_agnostic=False,
                 ratio_thr=0.8,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 fix_coder=dict(type='GVFixCoder'),
                 ratio_coder=dict(type='GVRatioCoder'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1./3., loss_weight=1.0),
                 loss_fix=dict(
                     type='SmoothL1Loss', beta=1./3., loss_weight=1.0),
                 loss_ratio=dict(
                     type='SmoothL1Loss', beta=1./3., loss_weight=16.0),
                ):
        super(GVBBoxHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.num_shared_fcs = num_shared_fcs
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.fc_out_channels = fc_out_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.ratio_thr = ratio_thr
        self.fp16_enabled = False
        self.start_bbox_type = 'hbb'
        self.end_bbox_type = 'poly'

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.fix_coder = build_bbox_coder(fix_coder)
        self.ratio_coder = build_bbox_coder(ratio_coder)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_fix = build_loss(loss_fix)
        self.loss_ratio = build_loss(loss_ratio)

        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area

        self.shared_fcs = nn.ModuleList()
        for i in range(self.num_shared_fcs):
            fc_in_channels = (
                in_channels if i == 0 else self.fc_out_channels)
            self.shared_fcs.append(
                nn.Linear(fc_in_channels, self.fc_out_channels))

        last_dim = in_channels if self.num_shared_fcs == 0 \
                else self.fc_out_channels
        self.fc_cls = nn.Linear(last_dim, self.num_classes + 1)
        out_dim_reg = (4 if self.reg_class_agnostic else 4*self.num_classes)
        self.fc_reg = nn.Linear(last_dim, out_dim_reg)
        out_dim_fix = (4 if self.reg_class_agnostic else 4*self.num_classes)
        self.fc_fix = nn.Linear(last_dim, out_dim_fix)
        out_dim_ratio = (1 if self.reg_class_agnostic else self.num_classes)
        self.fc_ratio = nn.Linear(last_dim, out_dim_ratio)

    def init_weights(self):
        for m in self.shared_fcs:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls.bias, 0)
        nn.init.normal_(self.fc_reg.weight, 0, 0.001)
        nn.init.constant_(self.fc_reg.bias, 0)
        nn.init.normal_(self.fc_fix.weight, 0, 0.001)
        nn.init.constant_(self.fc_fix.bias, 0)
        nn.init.normal_(self.fc_ratio.weight, 0, 0.001)
        nn.init.constant_(self.fc_ratio.bias, 0)

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.flatten(1)

        for fc in self.shared_fcs:
            x = self.relu(fc(x))

        cls_score = self.fc_cls(x)
        bbox_pred = self.fc_reg(x)
        fix_pred = torch.sigmoid(self.fc_fix(x))
        ratio_pred = torch.sigmoid(self.fc_ratio(x))
        return cls_score, bbox_pred, fix_pred, ratio_pred

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        fix_targets = pos_bboxes.new_zeros(num_samples, 4)
        fix_weights = pos_bboxes.new_zeros(num_samples, 4)
        ratio_targets = pos_bboxes.new_zeros(num_samples, 1)
        ratio_weights = pos_bboxes.new_zeros(num_samples, 1)

        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            pos_bbox_targets = self.bbox_coder.encode(
                pos_bboxes, bbox2type(pos_gt_bboxes, 'hbb'))
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1

            pos_fix_targets = self.fix_coder.encode(
                bbox2type(pos_gt_bboxes, 'poly'))
            fix_targets[:num_pos, :] = pos_fix_targets
            fix_weights[:num_pos, :] = 1

            pos_ratio_targets = self.ratio_coder.encode(
                bbox2type(pos_gt_bboxes, 'poly'))
            ratio_targets[:num_pos, :] = pos_ratio_targets
            ratio_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights, fix_targets,
                fix_weights, ratio_targets, ratio_weights)

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        outputs = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)
        (labels, label_weights, bbox_targets, bbox_weights, fix_targets,
         fix_weights, ratio_targets, ratio_weights) = outputs

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            fix_targets = torch.cat(fix_targets, 0)
            fix_weights = torch.cat(fix_weights, 0)
            ratio_targets = torch.cat(ratio_targets, 0)
            ratio_weights = torch.cat(ratio_weights, 0)
        return (labels, label_weights, bbox_targets, bbox_weights, fix_targets,
                fix_weights, ratio_targets, ratio_weights)

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'fix_pred', 'ratio_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             fix_pred,
             ratio_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             fix_targets,
             fix_weights,
             ratio_targets,
             ratio_weights,
             reduction_override=None):
        losses = dict()
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
        if cls_score.numel() > 0:
            losses['loss_cls'] = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc'] = accuracy(cls_score, labels)

        bg_class_ind = self.num_classes
        # 0~self.num_classes-1 are FG, self.num_classes is BG
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        # do not perform bounding box regression for BG anymore.
        if pos_inds.any():
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
            else:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), -1,
                    4)[pos_inds.type(torch.bool),
                       labels[pos_inds.type(torch.bool)]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds.type(torch.bool)],
                bbox_weights[pos_inds.type(torch.bool)],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        else:
            losses['loss_bbox'] = bbox_pred.sum() * 0

        if pos_inds.any():
            if self.reg_class_agnostic:
                pos_fix_pred = fix_pred.view(
                    fix_pred.size(0), 4)[pos_inds.type(torch.bool)]
            else:
                pos_fix_pred = fix_pred.view(
                    fix_pred.size(0), -1,
                    4)[pos_inds.type(torch.bool),
                       labels[pos_inds.type(torch.bool)]]
            losses['loss_fix'] = self.loss_fix(
                pos_fix_pred,
                fix_targets[pos_inds.type(torch.bool)],
                fix_weights[pos_inds.type(torch.bool)],
                avg_factor=fix_targets.size(0),
                reduction_override=reduction_override)
        else:
            losses['loss_fix'] = fix_pred.sum() * 0

        if pos_inds.any():
            if self.reg_class_agnostic:
                pos_ratio_pred = ratio_pred.view(
                    ratio_pred.size(0), 1)[pos_inds.type(torch.bool)]
            else:
                pos_ratio_pred = ratio_pred.view(
                    ratio_pred.size(0), -1,
                    1)[pos_inds.type(torch.bool),
                       labels[pos_inds.type(torch.bool)]]
            losses['loss_ratio'] = self.loss_ratio(
                pos_ratio_pred,
                ratio_targets[pos_inds.type(torch.bool)],
                ratio_weights[pos_inds.type(torch.bool)],
                avg_factor=ratio_targets.size(0),
                reduction_override=reduction_override)
        else:
            losses['loss_ratio'] = ratio_pred.sum() * 0

        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'fix_pred', 'ratio_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   fix_pred,
                   ratio_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1)
        bboxes = self.bbox_coder.decode(
            rois[:, 1:], bbox_pred, max_shape=img_shape)
        polys = self.fix_coder.decode(bboxes, fix_pred)

        bboxes = bboxes.view(*ratio_pred.size(), 4)
        polys = polys.view(*ratio_pred.size(), 8)
        polys[ratio_pred > self.ratio_thr] = \
                hbb2poly(bboxes[ratio_pred > self.ratio_thr])

        if rescale:
            if isinstance(scale_factor, float):
                scale_factor = [scale_factor for _ in range(4)]
            scale_factor = bboxes.new_tensor(scale_factor)
            polys /= scale_factor.repeat(2)
        polys = polys.view(polys.size(0), -1)

        if cfg is None:
            return polys, scores
        else:
            det_bboxes, det_labels = \
                    multiclass_arb_nms(polys, scores, cfg.score_thr,
                                       cfg.nms, cfg.max_per_img,
                                       bbox_type='poly')
            return det_bboxes, det_labels

    def refine_bboxes(self, rois, labels, bbox_preds, fix_preds, ratio_preds,
                      pos_is_gts, img_metas):
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            fix_pred_ = fix_preds[inds]
            ratio_pred_ = ratio_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           fix_pred_, ratio_pred_, img_meta_)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', 'fix_pred', 'ratio_pred'))
    def regress_by_class(self, rois, label, bbox_pred, fix_pred, ratio_pred, img_meta):
        assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)

        if not self.reg_class_agnostic:
            ratio_pred = torch.gather(ratio_pred, 1, label[:, None])
            label = label * 4
            inds = torch.stack([label + i for i in range(4)], 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
            fix_pred = torch.gather(fix_pred, 1, inds)
        assert bbox_pred.size(1) == 4
        assert fix_pred.size(1) == 4

        if rois.size(1) == 4:
            bboxes = self.bbox_coder.decode(
                rois, bbox_pred, max_shape=img_meta['img_shape'])
            new_rois = self.fix_coder.decode(bboxes, fix_pred)
            ratio_pred = ratio_pred.squeeze(1)
            new_rois[ratio_pred > self.ratio_thr] = hbb2poly(bboxes)
        else:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_meta['img_shape'])
            polys = self.fix_coder.decode(bboxes, fix_pred)
            ratio_pred = ratio_pred.squeeze(1)
            polys[ratio_pred > self.ratio_thr] = hbb2poly(bboxes)
            new_rois = torch.cat((rois[:, [0]], polys), dim=1)
        return new_rois
