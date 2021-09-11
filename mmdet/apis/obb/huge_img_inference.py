import mmcv
import json
import torch
import warnings
import numpy as np
import os.path as osp
import BboxToolkit as bt
from math import ceil
from itertools import product

from mmcv.parallel import collate, scatter

from mmdet.datasets.pipelines import Compose
from mmdet.ops import RoIAlign, RoIPool, nms, nms_rotated


def get_windows(img_W, img_H, sizes, steps, in_rate_thr=0.6):
    assert 1 >= in_rate_thr >= 0, 'The `in_rate_thr` should lie in 0~1'
    windows = []
    for size, step in zip(sizes, steps):
        assert size > step, 'Size should large than step'

        x_num = 1 if img_W <= size else ceil((img_W - size) / step + 1)
        x_start = [step * i for i in range(x_num)]
        if len(x_start) > 1 and x_start[-1] + size > img_W:
            x_start[-1] = img_W - size

        y_num = 1 if img_H <= size else ceil((img_H - size) / step + 1)
        y_start = [step * i for i in range(y_num)]
        if len(y_start) > 1 and y_start[-1] + size > img_H:
            y_start[-1] = img_H - size

        start = np.array(list(product(x_start, y_start)), dtype=np.int64)
        windows.append(np.concatenate([start, start + size], axis=1))
    windows = np.concatenate(windows, axis=0)

    img_contour = np.array([[0, 0, img_W, img_H]])
    win_iofs = bt.bbox_overlaps(windows, img_contour, mode='iof').reshape(-1)
    if not np.any(win_iofs >= in_rate_thr):
        win_iofs[abs(win_iofs - win_iofs.max()) < 0.01] = 1

    return windows[win_iofs >= in_rate_thr]


class LoadPatch(object):

    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None

        img = mmcv.imread(results['img'])
        x_start, y_start, x_stop, y_stop = results['patch_win']
        ph_h = y_stop - y_start
        ph_w = x_stop - y_start

        patch = img[y_start:y_stop, x_start:x_stop]
        if ph_h > patch.shape[0] or ph_w > patch.shape[1]:
            patch = mmcv.impad(patch, (ph_h, ph_w), self.fill)

        results['img'] = patch
        results['img_fields'] = ['img']
        results['img_shape'] = patch.shape
        results['ori_shape'] = patch.shape
        return results


def parse_split_cfg(split_cfg):
    if isinstance(split_cfg, str):
        assert osp.isfile(split_cfg) and split_cfg[-5:] == '.json'
        with open(split_cfg, 'r') as f:
            split_cfg = json.load(f)

    if isinstance(split_cfg, dict):
        sizes = split_cfg['sizes']
        gaps = split_cfg['gaps']
        rates = split_cfg['rates'] if 'rates' in split_cfg \
                else [1.0]
    elif isinstance(split_cfg, list):
        if len(split_cfg) == 2:
            sizes, gaps = split_cfg
        elif len(split_cfg) == 3:
            size, gaps, rates = split_cfg

    sizes = sizes if isinstance(sizes, list) else [sizes]
    gaps = gaps if isinstance(gaps, list) else [gaps]
    rates = rates if isinstance(rates, list) else [rates]
    _sizes, _steps = [], []
    for size, gap in zip(sizes, gaps):
        for rate in rates:
            _sizes.append(round(size / rate))
            _steps.append(round((size - gap) / rate))
    return _sizes, _steps


def inference_detector_huge_image(model, img, split_cfg, merge_cfg):
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadPatch()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # assert model
    is_cuda = next(model.parameters()).is_cuda
    if not is_cuda:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
    # generate patch windows
    img = mmcv.imread(img)
    height, width = img.shape[:2]
    sizes, steps = parse_split_cfg(split_cfg)
    windows = get_windows(width, height, sizes, steps)
    # detection loop
    results = []
    prog_bar = mmcv.ProgressBar(len(windows))
    for win in windows:
        data = dict(img=img)
        data['patch_win'] = win.tolist()
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            # just get the actual data from DataContainer
            data['img_metas'] = data['img_metas'][0].data

        # forward the model
        with torch.no_grad():
            results.append(model(return_loss=False, rescale=True, **data))
        prog_bar.update()
    # merge results
    print()
    print('Merge patch results!!')
    results = merge_patch_results(results, windows, merge_cfg)
    return results


def merge_patch_results(results, windows, nms_cfg):
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'BT_nms')
    try:
        nms_op = getattr(nms_rotated, nms_type)
    except AttributeError:
        nms_op = getattr(nms, nms_type)

    _results = []
    for _cls_result in zip(*results):
        cls_result = []
        for dets, win in zip(_cls_result, windows):
            bboxes, scores = dets[:, :-1], dets[:, [-1]]
            x_start, y_start = win[:2]
            bboxes = bt.translate(bboxes, x_start, y_start)
            cls_result.append(np.concatenate([bboxes, scores], axis=1))

        cls_result = np.concatenate(cls_result, axis=0)
        _result, _ = nms_op(cls_result, **nms_cfg_)
        _results.append(_result)
    return _results
