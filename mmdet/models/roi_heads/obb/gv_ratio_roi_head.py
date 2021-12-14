import torch
import torch.nn as nn
import numpy as np

from .obb_standard_roi_head import OBBStandardRoIHead
from mmdet.core import (arb2roi, arb2result, arb_mapping, merge_rotate_aug_arb,
                        multiclass_arb_nms)
from mmdet.models.builder import HEADS


@HEADS.register_module()
class GVRatioRoIHead(OBBStandardRoIHead):

    def forward_dummy(self, x, proposals):
        outs = ()
        rois = arb2roi([proposals], self.bbox_head.start_bbox_type)
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'],
                           bbox_results['fix_pred'],
                           bbox_results['ratio_pred'])
        return outs

    def _bbox_forward(self, x, rois):
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        outputs = self.bbox_head(bbox_feats)
        cls_score, bbox_pred, fix_pred, ratio_pred = outputs
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred,
            fix_pred=fix_pred, ratio_pred=ratio_pred,
            bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        rois = arb2roi([res.bboxes for res in sampling_results],
                       bbox_type=self.bbox_head.start_bbox_type)
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'],
                                        bbox_results['fix_pred'],
                                        bbox_results['ratio_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        rois = arb2roi(proposal_list, bbox_type=self.bbox_head.start_bbox_type)
        bbox_results = self._bbox_forward(x, rois)
        img_shape = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_bboxes(
            rois,
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            bbox_results['fix_pred'],
            bbox_results['ratio_pred'],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=self.test_cfg)

        bbox_results = arb2result(det_bboxes, det_labels, self.bbox_head.num_classes,
                                  bbox_type=self.bbox_head.end_bbox_type)
        return bbox_results

    def aug_test(self, feats, proposal_list, img_metas, rescale=False):
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
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
            # TODO more flexible
            proposals = arb_mapping(proposal_list[0][:, :-1], img_shape, scale_factor,
                                    h_flip, v_flip, rotate_after_flip, angle, matrix,
                                    bbox_type=self.bbox_head.start_bbox_type)
            rois = arb2roi([proposals], bbox_type=self.bbox_head.start_bbox_type)
            # recompute feature maps to save GPU memory
            bbox_results = self._bbox_forward(x, rois)
            bboxes, scores = self.bbox_head.get_bboxes(
                rois,
                bbox_results['cls_score'],
                bbox_results['bbox_pred'],
                bbox_results['fix_pred'],
                bbox_results['ratio_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_rotate_aug_arb(
            aug_bboxes, aug_scores, img_metas, self.test_cfg, merge_type='avg',
            bbox_type=self.bbox_head.end_bbox_type)
        det_bboxes, det_labels = multiclass_arb_nms(merged_bboxes, merged_scores,
                                                    self.test_cfg.score_thr,
                                                    self.test_cfg.nms,
                                                    self.test_cfg.max_per_img,
                                                    bbox_type=self.bbox_head.end_bbox_type)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            scale_factor = _det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
            _det_bboxes[:, :8] *= scale_factor.repeat(2)

        bbox_result = arb2result(
            det_bboxes, det_labels, self.bbox_head.num_classes,
            bbox_type='poly')
        return bbox_result
