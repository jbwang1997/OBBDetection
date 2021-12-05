import mmcv
import warnings
import numpy as np
import BboxToolkit as bt

from ..base import BaseDetector


class OBBBaseDetector(BaseDetector):

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    colors='green',
                    thickness=1.,
                    font_size=10,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):

        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = segms[i]
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        bboxes, scores = bboxes[:, :-1], bboxes[:, -1]
        bboxes = bboxes[scores > score_thr]
        labels = labels[scores > score_thr]
        scores = scores[scores > score_thr]
        img = bt.imshow_bboxes(
            img, bboxes, labels, scores,
            class_names=self.CLASSES,
            colors=colors,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img
