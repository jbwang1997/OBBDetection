import cv2
import numpy as np
import BboxToolkit as bt
import pycocotools.mask as maskUtils

from mmdet.core import PolygonMasks, BitmapMasks

pi = 3.141592


def bbox2mask(bboxes, w, h, mask_type='polygon'):
    polys = bt.bbox2type(bboxes, 'poly')
    assert mask_type in ['polygon', 'bitmap']
    if mask_type == 'bitmap':
        masks = []
        for poly in polys:
            rles = maskUtils.frPyObjects([poly.tolist()], h, w)
            masks.append(maskUtils.decode(rles[0]))
        gt_masks = BitmapMasks(masks, h, w)

    else:
        gt_masks = PolygonMasks([[poly] for poly in polys], h, w)
    return gt_masks


def switch_mask_type(masks, mtype='bitmap'):
    if isinstance(masks, PolygonMasks) and mtype == 'bitmap':
        width, height = masks.width, masks.height
        bitmap_masks = []
        for poly_per_obj in masks.masks:
            rles = maskUtils.frPyObjects(poly_per_obj, height, width)
            rle = maskUtils.merge(rles)
            bitmap_masks.append(maskUtils.decode(rle).astype(np.uint8))
        masks = BitmapMasks(bitmap_masks, height, width)
    elif isinstance(masks, BitmapMasks) and mtype == 'polygon':
        width, height = masks.width, masks.height
        polygons = []
        for bitmask in masks.masks:
            try:
                contours, _ = cv2.findContours(
                    bitmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except ValueError:
                _, contours, _ = cv2.findContours(
                    bitmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons.append(list(contours))
        masks = PolygonMasks(polygons, width, height)
    return masks


def rotate_polygonmask(masks, matrix, width, height):
    if len(masks) == 0:
        return masks

    points, sections, instances = [], [], []
    for i, polys_per_obj in enumerate(masks):
        for j, poly in enumerate(polys_per_obj):
            poly_points = poly.reshape(-1, 2)
            num_points = poly_points.shape[0]

            points.append(poly_points)
            sections.append(np.full((num_points, ), j))
            instances.append(np.full((num_points, ), i))
    points = np.concatenate(points, axis=0)
    sections = np.concatenate(sections, axis=0)
    instances = np.concatenate(instances, axis=0)

    points = cv2.transform(points[:, None, :], matrix)[:, 0, :]
    warpped_polygons = []
    for i in range(len(masks)):
        _points = points[instances == i]
        _sections = sections[instances == i]
        warpped_polygons.append(
            [_points[_sections == j].reshape(-1)
             for j in np.unique(_sections)])
    return PolygonMasks(warpped_polygons, height, width)


def polymask2hbb(masks):
    hbbs = []
    for mask in masks:
        all_mask_points = np.concatenate(mask, axis=0).reshape(-1, 2)
        min_points = all_mask_points.min(axis=0)
        max_points = all_mask_points.max(axis=0)
        hbbs.append(np.concatenate([min_points, max_points], axis=0))

    hbbs = np.array(hbbs, dtype=np.float32) if hbbs else \
            np.zeros((0, 4), dtype=np.float32)
    return hbbs


def polymask2obb(masks):
    obbs = []
    for mask in masks:
        all_mask_points = np.concatenate(mask, axis=0).reshape(-1, 2)
        all_mask_points = all_mask_points.astype(np.float32)
        (x, y), (w, h), angle = cv2.minAreaRect(all_mask_points)
        angle = -angle
        theta = angle / 180 * pi
        obbs.append([x, y, w, h, theta])

    if not obbs:
        obbs = np.zeros((0, 5), dtype=np.float32)
    else:
        obbs = np.array(obbs, dtype=np.float32)
    obbs = bt.regular_obb(obbs)
    return obbs


def polymask2poly(masks):
    polys = []
    for mask in masks:
        all_mask_points = np.concatenate(mask, axis=0)[None, :]
        if all_mask_points.size != 8:
            all_mask_points = bt.bbox2type(all_mask_points, 'obb')
            all_mask_points = bt.bbox2type(all_mask_points, 'poly')
        polys.append(all_mask_points)

    if not polys:
        polys = np.zeros((0, 8), dtype=np.float32)
    else:
        polys = np.concatenate(polys, axis=0)
    return polys


def bitmapmask2hbb(masks):
    if len(masks) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    bitmaps = masks.masks
    height, width = masks.height, masks.width
    num = bitmaps.shape[0]

    x, y = np.arange(width), np.arange(height)
    xx, yy = np.meshgrid(x, y)
    coors = np.stack([xx, yy], axis=-1)
    coors = coors[None, ...].repeat(num, axis=0)

    coors_ = coors.copy()
    coors_[bitmaps == 0] = -1
    max_points = np.max(coors_, axis=(1, 2)) + 1
    coors_ = coors.copy()
    coors_[bitmaps == 0] = 99999
    min_points = np.min(coors_, axis=(1, 2))

    hbbs = np.concatenate([min_points, max_points], axis=1)
    hbbs = hbbs.astype(np.float32)
    return hbbs


def bitmapmask2obb(masks):
    if len(masks) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    height, width = masks.height, masks.width
    x, y = np.arange(width), np.arange(height)
    xx, yy = np.meshgrid(x, y)
    coors = np.stack([xx, yy], axis=-1)
    coors = coors.astype(np.float32)

    obbs = []
    for mask in masks:
        points = coors[mask == 1]
        (x, y), (w, h), angle = cv2.minAreaRect(points)
        angle = -angle
        theta = angle / 180 * pi
        obbs.append([x, y, w, h, theta])

    obbs = np.array(obbs, dtype=np.float32)
    obbs = bt.regular_obb(obbs)
    return obbs


def bitmapmask2poly(masks):
    if len(masks) == 0:
        return np.zeros((0, 8), dtype=np.float32)

    height, width = masks.height, masks.width
    x, y = np.arange(width), np.arange(height)
    xx, yy = np.meshgrid(x, y)
    coors = np.stack([xx, yy], axis=-1)
    coors = coors.astype(np.float32)

    obbs = []
    for mask in masks:
        points = coors[mask == 1]
        (x, y), (w, h), angle = cv2.minAreaRect(points)
        angle = -angle
        theta = angle / 180 * pi
        obbs.append([x, y, w, h, theta])

    obbs = np.array(obbs, dtype=np.float32)
    return bt.bbox2type(obbs, 'poly')


def mask2bbox(masks, btype):
    if isinstance(masks, PolygonMasks):
        tran_func = bt.choice_by_type(polymask2hbb,
                                      polymask2obb,
                                      polymask2poly,
                                      btype)
    elif isinstance(masks, BitmapMasks):
        tran_func = bt.choice_by_type(bitmapmask2hbb,
                                      bitmapmask2obb,
                                      bitmapmask2poly,
                                      btype)
    else:
        raise NotImplementedError
    return tran_func(masks)
