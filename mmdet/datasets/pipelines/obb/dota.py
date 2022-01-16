import numpy as np
import BboxToolkit as bt
import cv2

from mmdet.core import PolygonMasks, BitmapMasks
from mmdet.datasets.builder import PIPELINES
from .misc import mask2bbox


@PIPELINES.register_module()
class LoadDOTASpecialInfo(object):

    def __init__(self,
                 img_keys=dict(gsd='gsd'),
                 ann_keys=dict(diffs='diffs', trunc='trunc'),
                 split_keys=dict(
                     split_sizes='sizes',
                     split_rates='rates',
                     split_gaps='gaps')):
        self.img_keys = img_keys
        self.ann_keys = ann_keys
        self.split_keys = split_keys

    def __call__(self, results):
        for res_key, img_key in self.img_keys.items():
            results[res_key] = results['img_info'][img_key]
        for res_key, split_key in self.split_keys.items():
            results[res_key] = results['split_info'][split_key]
        results['aligned_fields'] = []
        for res_key, ann_key in self.ann_keys.items():
            results[res_key] = results['ann_info'][ann_key]
            results['aligned_fields'].append(res_key)
        return results

@PIPELINES.register_module()
class DOTASpecialIgnore(object):

    def __init__(self,
                 ignore_diff=False,
                 ignore_truncated=False,
                 ignore_size=None,
                 ignore_real_scales=None):
        self.ignore_diff = ignore_diff
        self.ignore_truncated = ignore_truncated
        self.ignore_size = ignore_size
        self.ignore_real_scales = ignore_real_scales

    def __call__(self, results):
        for k in ['gt_bboxes', 'gt_masks', 'gt_labels']:
            if k in results:
                num_objs = len(results[k])
                break
        else:
            return results

        ignore = np.zeros((num_objs, ), dtype=np.bool)
        if self.ignore_diff:
            assert 'diffs' in results
            diffs = results['diffs']
            ignore[diffs == 1] = True

        if self.ignore_truncated:
            assert 'trunc' in results
            trunc = results['trunc']
            ignore[trunc == 1] = True

        if self.ignore_size:
            bboxes = results['gt_bboxes']
            wh = bboxes[:, 2:] - bboxes[:, :2]
            ignore[np.min(wh, axis=1) < self.ignore_size] = True

        if self.ignore_real_scales:
            assert len(self.ignore_real_scales) == (len(results['split_sizes']) *
                                                    len(results['split_rates']))
            polys = mask2bbox(results['gt_masks'], 'poly')
            if 'scale_factor' in results:
                scale_factor = np.tile(results['scale_factor'], 2)
                polys = polys / scale_factor
            bbox_scales = np.sqrt(bt.bbox_areas(polys))

            split_sizes=[]
            for rate in results['split_rates']:
                split_sizes += [int(size / rate) for size in results['split_sizes']]
            img_scale = results['img_info']['width']
            scale_ratio = np.array(split_sizes) / img_scale
            inds = np.argmin(abs(np.log(scale_ratio)))
            min_scale, max_scale = self.ignore_real_scales[inds]
            if min_scale is None:
                min_scale = 0
            if max_scale is None:
                max_scale = np.inf
            ignore[bbox_scales < min_scale] = True
            ignore[bbox_scales > max_scale] = True

        if 'gt_bboxes' in results:
            bboxes = results['gt_bboxes']
            gt_bboxes = bboxes[~ignore]
            gt_bboxes_ignore = bboxes[ignore]

            results['gt_bboxes'] = gt_bboxes
            results['gt_bboxes_ignore'] = gt_bboxes_ignore
            if 'gt_bboxes_ignore' not in results['bbox_fields']:
                results['bbox_fields'].append('gt_bboxes_ignore')

        if 'gt_masks' in results:
            gt_inds = np.nonzero(~ignore)[0]
            ignore_inds = np.nonzero(ignore)[0]

            if isinstance(results['gt_masks'], PolygonMasks) \
               and len(gt_inds) == 0:
                height = results['gt_masks'].height
                width = results['gt_masks'].width
                gt_masks = PolygonMasks([], height, width)
            else:
                gt_masks = results['gt_masks'][gt_inds]

            if isinstance(results['gt_masks'], PolygonMasks) \
               and len(ignore_inds) == 0:
                height = results['gt_masks'].height
                width = results['gt_masks'].width
                gt_masks_ignore = PolygonMasks([], height, width)
            else:
                gt_masks_ignore = results['gt_masks'][ignore_inds]

            results['gt_masks'] = gt_masks
            results['gt_masks_ignore'] = gt_masks_ignore
            if 'gt_masks_ignore' not in results['mask_fields']:
                results['mask_fields'].append('gt_masks_ignore')

        if 'gt_labels' in results:
            results['gt_labels'] = results['gt_labels'][~ignore]

        for k in results.get('aligned_fields', []):
            results[k] = results[k][~ignore]

        return results
