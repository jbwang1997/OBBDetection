import torch
import torch.nn as nn
import numpy as np

from mmdet.core import (hbb_mapping, build_assigner,
                        build_sampler, merge_rotate_aug_arb,
                        multiclass_arb_nms)
from mmdet.core import arb2roi, arb2result
from mmdet.core import regular_obb, get_bbox_dim
from mmdet.models.builder import HEADS, build_head, build_roi_extractor
from .obb_base_roi_head import OBBBaseRoIHead


@HEADS.register_module()
class RoITransRoIHead(OBBBaseRoIHead):

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
                'Shared head is not supported in Cascade RCNN anymore'
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights

        super(RoITransRoIHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        self.bbox_roi_extractor = nn.ModuleList()
        self.bbox_head = nn.ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages

        last_type = ['hbb', 'obb', 'poly']
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))

            assert self.bbox_head[-1].start_bbox_type in last_type
            last_type = [self.bbox_head[-1].end_bbox_type]

    def init_assigner_sampler(self):
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for rcnn_train_cfg in self.train_cfg:
                self.bbox_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                self.bbox_sampler.append(build_sampler(rcnn_train_cfg.sampler, context=self))
        # sampler need context, init in train procession

    def init_weights(self, pretrained):
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()

    def forward_dummy(self, x, proposals):
        outs = ()
        for i in range(self.num_stages):
            start_bbox_type = self.bbox_head[i].start_bbox_type
            if start_bbox_type == 'hbb':
                rois = arb2roi([proposals], bbox_type='hbb')
            else:
                theta = proposals.new_zeros((proposals.size(0), 1))
                proposal_theta = torch.cat([proposals, theta], dim=1)
                rois = arb2roi([proposal_theta], bbox_type='obb')

            bbox_results = self._bbox_forward(i, x, rois)
            outs = outs + (bbox_results['cls_score'], bbox_results['bbox_pred'])
        return outs

    def _bbox_forward(self, stage, x, rois):
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg):
        rois = arb2roi([res.bboxes for res in sampling_results],
                       bbox_type=self.bbox_head[stage].start_bbox_type)
        bbox_results = self._bbox_forward(stage, x, rois)
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'], rois,
                                               *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_obboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_obboxes_ignore=None):
        assert gt_obboxes is not None
        losses = dict()
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]
            start_bbox_type = self.bbox_head[i].start_bbox_type
            end_bbox_type = self.bbox_head[i].end_bbox_type

            # assign gts and sample proposals
            sampling_results = []
            target_bboxes = gt_bboxes if start_bbox_type == 'hbb' else gt_obboxes
            target_bboxes_ignore = gt_bboxes_ignore \
                    if start_bbox_type == 'hbb' else gt_obboxes_ignore
            if self.with_bbox:
                bbox_assigner = self.bbox_assigner[i]
                bbox_sampler = self.bbox_sampler[i]
                num_imgs = len(img_metas)
                if target_bboxes_ignore is None:
                    target_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], target_bboxes[j],
                        target_bboxes_ignore[j], gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        target_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])

                    if start_bbox_type != end_bbox_type:
                        if gt_obboxes[j].numel() == 0:
                            sampling_result.pos_gt_bboxes = gt_obboxes[i].new_zeors(
                                (0, gt_obboxes[0].size(-1)))
                        else:
                            sampling_result.pos_gt_bboxes = \
                                    gt_obboxes[j][sampling_result.pos_assigned_gt_inds, :]

                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(i, x, sampling_results, target_bboxes,
                                                    gt_labels, rcnn_train_cfg)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                # bbox_targets is a tuple
                roi_labels = bbox_results['bbox_targets'][0]
                with torch.no_grad():
                    roi_labels = torch.where(
                        roi_labels == self.bbox_head[i].num_classes,
                        bbox_results['cls_score'][:, :-1].argmax(1),
                        roi_labels)
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)

        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        img_shape = img_metas[0]['img_shape']
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']

        # "ms" in variable names means multi-stage
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = arb2roi(proposal_list, bbox_type=self.bbox_head[0].start_bbox_type)
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)
            ms_scores.append(bbox_results['cls_score'])

            if i < self.num_stages - 1:
                bbox_label = bbox_results['cls_score'].argmax(dim=1)
                rois = self.bbox_head[i].regress_by_class(
                    rois, bbox_label, bbox_results['bbox_pred'], img_metas[0])

        score_weights = rcnn_test_cfg.score_weights
        assert len(score_weights) == len(ms_scores)
        cls_score = sum([w * s for w, s in zip(score_weights, ms_scores)]) \
                / sum(score_weights)

        det_bboxes, det_labels = self.bbox_head[-1].get_bboxes(
            rois,
            cls_score,
            bbox_results['bbox_pred'],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        results = arb2result(det_bboxes, det_labels, self.bbox_head[-1].num_classes,
                             bbox_type=self.bbox_head[-1].end_bbox_type)
        return results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(features, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            angle = img_meta[0].get('angle', 0)
            matrix = img_meta[0].get('matrix', np.eye(3))
            rotate_after_flip = img_meta[0].get('rotate_after_flip', True)
            if 'flip' in img_meta[0]:
                direction = img_meta[0]['flip_direction']
                h_flip = img_meta[0]['flip'] if direction == 'horizontal' else False
                v_flip = img_meta[0]['flip'] if direction == 'vertical' else False
            else:
                h_flip, v_flip = img_meta[0]['h_flip'], img_meta[0]['v_flip']

            proposals = hbb_mapping(proposal_list[0][:, :4], img_shape, scale_factor,
                                    h_flip, v_flip, rotate_after_flip, angle, matrix)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = arb2roi([proposals], bbox_type=self.bbox_head[0].start_bbox_type)
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'].argmax(dim=1)
                    rois = self.bbox_head[i].regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            score_weights = rcnn_test_cfg.score_weights
            assert len(score_weights) == len(ms_scores)
            cls_score = sum([w * s for w, s in zip(score_weights, ms_scores)]) \
                    / sum(score_weights)
            cls_score = ms_scores[0]

            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        end_bbox_type = self.bbox_head[-1].end_bbox_type
        merged_bboxes, merged_scores = merge_rotate_aug_arb(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg,
            merge_type='avg', bbox_type=end_bbox_type)
        det_bboxes, det_labels = multiclass_arb_nms(
            merged_bboxes, merged_scores, rcnn_test_cfg.score_thr,
            rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img,
            bbox_type=end_bbox_type)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            scale_factor = _det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
            if end_bbox_type == 'poly':
                _det_bboxes[:, :8] *= scale_factor.repeat(2)
            else:
                _det_bboxes[:, :4] *= scale_factor

        bbox_result = arb2result(
            det_bboxes, det_labels, self.bbox_head[-1].num_classes,
            bbox_type=self.bbox_head[-1].end_bbox_type)
        return bbox_result
