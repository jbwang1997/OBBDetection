import logging
import sys
import numpy as np

import torch

from mmdet.core import (arb2roi, arb_mapping, merge_rotate_aug_arb,
                        get_bbox_type, multiclass_arb_nms)

logger = logging.getLogger(__name__)

if sys.version_info >= (3, 7):
    from mmdet.utils.contextmanagers import completed


class OBBoxTestMixin(object):

    if sys.version_info >= (3, 7):

        async def async_test_bboxes(self,
                                    x,
                                    img_metas,
                                    proposals,
                                    rcnn_test_cfg,
                                    rescale=False,
                                    bbox_semaphore=None,
                                    global_lock=None):
            """Asynchronized test for box head without augmentation."""
            rois = arb2roi(proposals, bbox_type=self.bbox_head.start_bbox_type)
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            sleep_interval = rcnn_test_cfg.get('async_sleep_interval', 0.017)

            async with completed(
                    __name__, 'bbox_head_forward',
                    sleep_interval=sleep_interval):
                cls_score, bbox_pred = self.bbox_head(roi_feats)

            img_shape = img_metas[0]['img_shape']
            scale_factor = img_metas[0]['scale_factor']
            det_bboxes, det_labels = self.bbox_head.get_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=rcnn_test_cfg)
            return det_bboxes, det_labels

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = arb2roi(proposals, bbox_type=self.bbox_head.start_bbox_type)
        bbox_results = self._bbox_forward(x, rois)
        img_shape = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_bboxes(
            rois,
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        """Test det bboxes with test time augmentation."""
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
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_rotate_aug_arb(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg, merge_type='avg',
            bbox_type=self.bbox_head.end_bbox_type)
        det_bboxes, det_labels = multiclass_arb_nms(merged_bboxes, merged_scores,
                                                    rcnn_test_cfg.score_thr,
                                                    rcnn_test_cfg.nms,
                                                    rcnn_test_cfg.max_per_img,
                                                    bbox_type=self.bbox_head.end_bbox_type)
        return det_bboxes, det_labels
