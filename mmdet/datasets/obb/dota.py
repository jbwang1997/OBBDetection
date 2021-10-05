import BboxToolkit as bt

import os
import cv2
import time
import mmcv
import numpy as np
import os.path as osp
# from multiprocessing import Pool

from functools import partial
from mmdet.core import eval_arb_map, eval_arb_recalls
from mmdet.ops.nms_rotated import obb_nms, poly_nms, BT_nms
from mmdet.ops.nms import nms
from collections import defaultdict
from ..builder import DATASETS
from ..custom import CustomDataset


@DATASETS.register_module()
class DOTADataset(CustomDataset):

    def __init__(self,
                 task,
                 *args,
                 **kwargs):
        assert task in ['Task1', 'Task2']
        self.task = task
        super(DOTADataset, self).__init__(*args, **kwargs)

    @classmethod
    def get_classes(cls, classes=None):
        if classes is None:
            cls.custom_classes = False
            return None

        cls.custom_classes = True
        return bt.get_classes(classes)

    def load_annotations(self, ann_file):
        split_config = osp.join(ann_file, 'split_config.json')
        self.split_info = mmcv.load(split_config)

        ori_annfile = osp.join(ann_file, 'ori_annfile.pkl')
        self.ori_infos = mmcv.load(ori_annfile)['content']

        patch_annfile = osp.join(ann_file, 'patch_annfile.pkl')
        patch_dict = mmcv.load(patch_annfile)
        cls, contents = patch_dict['cls'], patch_dict['content']
        self.ori_CLASSES = cls
        if self.CLASSES is None:
            self.CLASSES = cls

        if not self.test_mode:
            data_infos = []
            for content in contents:
                if len(content['ann']['bboxes']) != 0:
                    data_infos.append(content)
        else:
            data_infos = contents
        return data_infos

    def get_subset_by_classes(self):
        bt.change_cls_order(self.data_infos, self.ori_CLASSES, self.CLASSES)
        return self.data_infos

    def pre_pipeline(self, results):
        results['split_info'] = self.split_info
        results['cls'] = self.CLASSES
        super().pre_pipeline(results)

    def format_results(self,
                       results,
                       with_merge=True,
                       ign_scale_ranges=None,
                       iou_thr=0.5,
                       nproc=4,
                       save_dir=None,
                       **kwargs):
        nproc = min(nproc, os.cpu_count())
        task = self.task
        if mmcv.is_list_of(results, tuple):
            dets, segments = results
            if task == 'Task1':
                dets = _list_mask_2_obb(dets, segments)
        else:
            dets = results

        if not with_merge:
            results = [(data_info['id'], result)
                       for data_info, result in zip(self.data_infos, results)]
            if save_dir is not None:
                id_list, dets_list = zip(*results)
                bt.save_dota_submission(save_dir, id_list, dets_list, task, self.CLASSES)
            return results

        print('\nMerging patch bboxes into full image!!!')
        start_time = time.time()
        if ign_scale_ranges is not None:
            assert len(ign_scale_ranges) == (len(self.split_info['rates']) *
                                             len(self.split_info['sizes']))
            split_sizes=[]
            for rate in self.split_info['rates']:
                split_sizes += [int(size / rate) for size in self.split_info['sizes']]

        collector = defaultdict(list)
        for data_info, result in zip(self.data_infos, dets):
            if ign_scale_ranges is not None:
                img_scale = data_info['width']
                scale_ratio = np.array(split_sizes) / img_scale
                inds = np.argmin(abs(np.log(scale_ratio)))

                min_scale, max_scale = ign_scale_ranges[inds]
                min_scale = 0 if min_scale is None else min_scale
                max_scale = np.inf if max_scale is None else max_scale

            x_start, y_start = data_info['x_start'], data_info['y_start']
            new_result = []
            for i, dets in enumerate(result):
                if ign_scale_ranges is not None:
                    bbox_scales = np.sqrt(bt.bbox_areas(dets[:, :-1]))
                    valid_inds = (bbox_scales > min_scale) & (bbox_scales < max_scale)
                    dets = dets[valid_inds]
                bboxes, scores = dets[:, :-1], dets[:, [-1]]
                bboxes = bt.translate(bboxes, x_start, y_start)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(np.concatenate(
                    [labels, bboxes, scores], axis=1))

            new_result = np.concatenate(new_result, axis=0)
            collector[data_info['ori_id']].append(new_result)

        merge_func = partial(
            _merge_func,
            CLASSES=self.CLASSES,
            iou_thr=iou_thr,
            task=task)
        if nproc <= 1:
            print('Single processing')
            merged_results = mmcv.track_iter_progress(
                (map(merge_func, collector.items()), len(collector)))
        else:
            print('Multiple processing')
            merged_results = mmcv.track_parallel_progress(
                merge_func, list(collector.items()), nproc)

        if save_dir is not None:
            id_list, dets_list = zip(*merged_results)
            bt.save_dota_submission(save_dir, id_list, dets_list, task, self.CLASSES)

        stop_time = time.time()
        print('Used time: %.1f s'%(stop_time - start_time))
        return merged_results

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 with_merge=True,
                 ign_diff=True,
                 ign_scale_ranges=None,
                 save_dir=None,
                 merge_iou_thr=0.1,
                 use_07_metric=True,
                 scale_ranges=None,
                 eval_iou_thr=[0.5],
                 proposal_nums=(2000, ),
                 nproc=10):
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        task = self.task

        eval_results = {}
        if metric == 'mAP':
            merged_results = self.format_results(
                results,
                nproc=nproc,
                with_merge=with_merge,
                ign_scale_ranges=ign_scale_ranges,
                iou_thr=merge_iou_thr,
                save_dir=save_dir)

            infos = self.ori_infos if with_merge else self.data_infos
            id_mapper = {ann['id']: i for i, ann in enumerate(infos)}
            det_results, annotations = [], []
            for k, v in merged_results:
                det_results.append(v)
                ann = infos[id_mapper[k]]['ann']
                gt_bboxes = ann['bboxes']
                gt_labels = ann['labels']
                diffs = ann.get(
                    'diffs', np.zeros((gt_bboxes.shape[0], ), dtype=np.int))

                if task == 'Task2':
                    gt_bboxes = bt.bbox2type(gt_bboxes, 'hbb')

                gt_ann = {}
                if ign_diff:
                    gt_ann['bboxes_ignore'] = gt_bboxes[diffs == 1]
                    gt_ann['labels_ignore'] = gt_labels[diffs == 1]
                    gt_bboxes = gt_bboxes[diffs == 0]
                    gt_labels = gt_labels[diffs == 0]
                gt_ann['bboxes'] = gt_bboxes
                gt_ann['labels'] = gt_labels
                annotations.append(gt_ann)

            print('\nStart calculate mAP!!!')
            print('Result is Only for reference,',
                  'final result is subject to DOTA_devkit')
            mean_ap, _ = eval_arb_map(
                det_results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=eval_iou_thr,
                use_07_metric=use_07_metric,
                dataset=self.CLASSES,
                logger=logger,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
        elif metric == 'recall':
            assert mmcv.is_list_of(results, np.ndarray)
            gt_bboxes = []
            for info in self.data_infos:
                bboxes = info['ann']['bboxes']
                if ign_diff:
                    diffs = info['ann'].get(
                        'diffs', np.zeros((bboxes.shape[0], ), dtype=np.int))
                    bboxes = bboxes[diffs == 0]
                gt_bboxes.append(bboxes)
            if isinstance(eval_iou_thr, float):
                eval_iou_thr = [eval_iou_thr]
            recalls = eval_arb_recalls(
                gt_bboxes, results, True, proposal_nums, eval_iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(eval_iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results


def _merge_func(info, CLASSES, iou_thr, task):
    img_id, label_dets = info
    label_dets = np.concatenate(label_dets, axis=0)
    labels, dets = label_dets[:, 0], label_dets[:, 1:]
    nms_ops = bt.choice_by_type(nms, obb_nms, BT_nms,
                                dets, with_score=True)

    big_img_results = []
    for i in range(len(CLASSES)):
        cls_dets = dets[labels == i]
        nms_dets, _ = nms_ops(cls_dets, iou_thr)

        if task == 'Task2':
            bboxes = bt.bbox2type(nms_dets[:, :-1], 'hbb')
            nms_dets = np.concatenate([bboxes, nms_dets[:, -1:]], axis=1)
        big_img_results.append(nms_dets)
    return img_id, big_img_results


def _list_mask_2_obb(dets, segments):
    new_dets = []
    for cls_dets, cls_segments in zip(dets, segments):
        new_cls_dets = []
        for ds, segs in zip(cls_dets, cls_segments):
            _, scores = ds[:, :-1], ds[:, -1]
            new_bboxes = []
            for seg in segs:
                try:
                    contours, _ = cv2.findContours(
                        seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                except ValueError:
                    _, contours, _ = cv2.findContours(
                        seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                max_contour = max(contours, key=len).reshape(1, -1)
                new_bboxes.append(bt.bbox2type(max_contour, 'obb'))

            new_bboxes = np.zeros((0, 5)) if not new_bboxes else \
                    np.concatenate(new_bboxes, axis=0)
            new_cls_dets.append(
                np.concatenate([new_bboxes, scores[:, None]], axis=1))
        new_dets.append(new_cls_dets)
    return new_dets
