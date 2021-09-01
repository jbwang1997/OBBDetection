import numpy as np
import torch

pi = 3.141592


def get_bbox_type(bboxes, with_score=False):
    dim = bboxes.size(-1)
    if with_score:
        dim -= 1

    if dim == 4:
        return 'hbb'
    if dim == 5:
        return 'obb'
    if dim  == 8:
        return 'poly'
    return 'notype'


def get_bbox_dim(bbox_type, with_score=False):
    if bbox_type == 'hbb':
        dim = 4
    elif bbox_type == 'obb':
        dim = 5
    elif bbox_type == 'poly':
        dim = 8
    else:
        raise ValueError(f"don't know {bbox_type} bbox dim")

    if with_score:
        dim += 1
    return dim


def get_bbox_areas(bboxes):
    btype = get_bbox_type(bboxes)
    if btype == 'hbb':
        wh = bboxes[..., 2:] - bboxes[..., :2]
        areas = wh[..., 0] * wh[..., 1]
    elif btype == 'obb':
        areas = bboxes[..., 2] * bboxes[..., 3]
    elif btype == 'poly':
        pts = bboxes.view(*bboxes.size()[:-1], 4, 2)
        roll_pts = torch.roll(pts, 1, dims=-2)
        xyxy = torch.sum(pts[..., 0] * roll_pts[..., 1] -
                         roll_pts[..., 0] * pts[..., 1], dim=-1)
        areas = 0.5 * torch.abs(xyxy)
    else:
        raise ValueError('The type of bboxes is notype')

    return areas


def choice_by_type(hbb_op, obb_op, poly_op, bboxes_or_type,
                   with_score=False):
    if isinstance(bboxes_or_type, torch.Tensor):
        bbox_type = get_bbox_type(bboxes_or_type, with_score)
    elif isinstance(bboxes_or_type, str):
        bbox_type = bboxes_or_type
    else:
        raise TypeError(f'need np.ndarray or str,',
                        f'but get {type(bboxes_or_type)}')

    if bbox_type == 'hbb':
        return hbb_op
    elif bbox_type == 'obb':
        return obb_op
    elif bbox_type == 'poly':
        return poly_op
    else:
        raise ValueError('notype bboxes is not suppert')


def arb2result(bboxes, labels, num_classes, bbox_type='hbb'):
    assert bbox_type in ['hbb', 'obb', 'poly']
    bbox_dim = get_bbox_dim(bbox_type, with_score=True)

    if bboxes.shape[0] == 0:
        return [np.zeros((0, bbox_dim), dtype=np.float32) for i in range(num_classes)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


def arb2roi(bbox_list, bbox_type='hbb'):
    assert bbox_type in ['hbb', 'obb', 'poly']
    bbox_dim = get_bbox_dim(bbox_type)

    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :bbox_dim]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, bbox_dim+1))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def distance2obb(points, distance, max_shape=None):
    distance, theta = distance.split([4, 1], dim=1)

    Cos, Sin = torch.cos(theta), torch.sin(theta)
    Matrix = torch.cat([Cos, Sin, -Sin, Cos], dim=1).reshape(-1, 2, 2)

    wh = distance[:, :2] + distance[:, 2:]
    offset_t = (distance[:, 2:] - distance[:, :2]) / 2
    offset_t = offset_t.unsqueeze(2)
    offset = torch.bmm(Matrix, offset_t).squeeze(2)
    ctr = points + offset

    obbs = torch.cat([ctr, wh, theta], dim=1)
    return regular_obb(obbs)


def regular_theta(theta, mode='180', start=-pi/2):
    assert mode in ['360', '180']
    cycle = 2 * pi if mode == '360' else pi

    theta = theta - start
    theta = theta % cycle
    return theta + start


def regular_obb(obboxes):
    x, y, w, h, theta = obboxes.unbind(dim=-1)
    w_regular = torch.where(w > h, w, h)
    h_regular = torch.where(w > h, h, w)
    theta_regular = torch.where(w > h, theta, theta+pi/2)
    theta_regular = regular_theta(theta_regular)
    return torch.stack([x, y, w_regular, h_regular, theta_regular], dim=-1)


def mintheta_obb(obboxes):
    x, y, w, h, theta = obboxes.unbind(dim=-1)
    theta1 = regular_theta(theta)
    theta2 = regular_theta(theta + pi/2)
    abs_theta1 = torch.abs(theta1)
    abs_theta2 = torch.abs(theta2)

    w_regular = torch.where(abs_theta1 < abs_theta2, w, h)
    h_regular = torch.where(abs_theta1 < abs_theta2, h, w)
    theta_regular = torch.where(abs_theta1 < abs_theta2, theta1, theta2)

    obboxes = torch.stack([x, y, w_regular, h_regular, theta_regular], dim=-1)
    return obboxes
