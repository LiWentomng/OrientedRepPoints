import mmcv
import numpy as np
import torch
import math
PI = np.pi

"""
transformation functions for oriented bounding boxes.
"""

def xyxy2xywht(roi_xyxy):
    xmin, ymin, xmax, ymax = roi_xyxy.split(1, dim=-1)
    
    ctr_x = (xmin + xmax) / 2
    ctr_y = (ymin + ymax) / 2
    width = (xmax - xmin + 1) 
    height = (ymax - ymin + 1)
    theta = torch.ones_like(ctr_x) * (-3.14 / 2)
    
    roi_xywht = torch.cat([ctr_x, ctr_y, width, height, theta], dim=1)
    return roi_xywht

def xywht2xyxyxyxy(roi_xywht, mode='opencv'):
    assert mode in ['opencv', 'rect'], "mode should in ['opencv', 'rect']"
    ctr_x, ctr_y, width, height, theta = roi_xywht.split(1, dim=-1)
    
    _theta = torch.abs(theta)
    sind = torch.sin(_theta)
    cosd = torch.cos(_theta)
    dw = width / 2
    dh = height / 2
    if mode == 'opencv':
        x1 = ctr_x + dw * cosd - dh * sind
        y1 = ctr_y - dw * sind - dh * cosd

        x2 = ctr_x - dw * cosd - dh * sind
        y2 = ctr_y + dw * sind - dh * cosd

        x3 = 2 * ctr_x - x1
        y3 = 2 * ctr_y - y1

        x4 = 2 * ctr_x - x2
        y4 = 2 * ctr_y - y2
    elif mode == 'rect':
        x1 = ctr_x + dh * cosd - dw * sind
        y1 = ctr_y - dh * sind - dw * cosd

        x2 = ctr_x - dh * cosd - dw * sind
        y2 = ctr_y + dh * sind - dw * cosd

        x3 = 2 * ctr_x - x1
        y3 = 2 * ctr_y - y1

        x4 = 2 * ctr_x - x2
        y4 = 2 * ctr_y - y2

    polygen = torch.cat((x1, y1, x2, y2, x3, y3, x4, y4), dim=-1)

    return polygen

def rbbox2delta(proposals, gt, means=[0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1]):
    assert proposals.size(0) == gt.size(0)

    proposals = proposals.float()
    gt = gt.float()
    px = proposals[..., 0]
    py = proposals[..., 1]
    pw = proposals[..., 2]
    ph = proposals[..., 3]
    pt = proposals[..., 4]

    gx = gt[..., 0]
    gy = gt[..., 1]
    gw = gt[..., 2]
    gh = gt[..., 3]
    gth = gt[..., 4]
    
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    dt = gth - pt
    
    deltas = torch.stack([dx, dy, dw, dh, dt], dim=-1)
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    return deltas

def delta2rbbox(rois,
               deltas,
               means=[0, 0, 0, 0, 0],
               stds=[1, 1, 1, 1, 1],
               max_shape=None,
               to_xyxyxyxy=False,
               to_mode="opencv",
               wh_ratio_clip=16 / 1000):
    
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    dt = denorm_deltas[:, 4::5]

    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    # Compute theta of each roi
    pt = torch.ones_like(px) * (-3.14 / 2)

    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    gt = pt + dt

    # Convert center-xy/width/height to top-left, bottom-right
    polygens_xywht = torch.cat([gx, gy,gw, gh, gt], dim=1)
    if to_xyxyxyxy:
        polygens = xywht2xyxyxyxy(polygens_xywht, mode=to_mode)
    # bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return polygens


def obbox2result(oboxes, labels, num_classes):
    """Convert oriented bounding box results to a list of numpy arrays.

    Args:
        oboxes (Tensor): shape (n, 9)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): obox results of each class
    """
    if oboxes.shape[0] == 0:
        return [
            np.zeros((0, 9), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        oboxes = oboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [oboxes[labels == i, :] for i in range(num_classes - 1)]


def rbox2poly(rrects):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    polys = []
    for rrect in rrects:
        x_ctr, y_ctr, width, height, angle = rrect[:5]
        tl_x, tl_y, br_x, br_y = -width/2, -height/2, width/2, height/2
        rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        poly = R.dot(rect)
        x0, x1, x2, x3 = poly[0, :4] + x_ctr
        y0, y1, y2, y3 = poly[1, :4] + y_ctr
        poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
        polys.append(poly)
    polys = np.array(polys)
    polys = get_best_begin_point(polys)
    return polys

def poly2rbox(polys):
    """
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rrect:[x_ctr,y_ctr,w,h,angle]
    """
    rrects = []
    for poly in polys:
        poly = np.array(poly[:8], dtype=np.float32)

        pt1 = (poly[0], poly[1])
        pt2 = (poly[2], poly[3])
        pt3 = (poly[4], poly[5])
        pt4 = (poly[6], poly[7])

        edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) +
                        (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
        edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) +
                        (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

        angle = 0
        width = 0
        height = 0

        if edge1 > edge2:
            width = edge1
            height = edge2
            angle = np.arctan2(
                np.float(pt2[1] - pt1[1]), np.float(pt2[0] - pt1[0]))
        elif edge2 >= edge1:
            width = edge2
            height = edge1
            angle = np.arctan2(
                np.float(pt4[1] - pt1[1]), np.float(pt4[0] - pt1[0]))

        angle = (angle + PI / 4) % PI - PI / 4

        x_ctr = np.float(pt1[0] + pt3[0]) / 2
        y_ctr = np.float(pt1[1] + pt3[1]) / 2
        rrect = np.array([x_ctr, y_ctr, width, height, angle])
        rrects.append(rrect)
    return np.array(rrects)

def cal_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def get_best_begin_point_single(coordinate):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) \
            + cal_line_length(combinate[i][1], dst_coordinate[1]) \
            + cal_line_length(combinate[i][2], dst_coordinate[2]) \
            + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
        # print("choose one direction!")
    return np.array(combinate[force_flag]).reshape(8)

def get_best_begin_point(coordinates):
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates

def obbox_flip(obboxes, img_shape, direction='horizontal'):
    """Flip obboxes horizontally. (poly bounding box)

    Args:
        obboxes (list[Tensor]): shape (..., 8*k)
        img_shape(tuple): (height, width)
    """
    assert obboxes.shape[-1] % 8 == 0
    flipped = obboxes.clone()

    if direction == 'horizontal':
        w = img_shape[1]
        flipped[..., 0::8] = w - obboxes[..., 0::8] - 1
        flipped[..., 2::8] = w - obboxes[..., 2::8] - 1
        flipped[..., 4::8] = w - obboxes[..., 4::8] - 1
        flipped[..., 6::8] = w - obboxes[..., 6::8] - 1
    elif direction == 'vertical':
        h = img_shape[0]
        flipped[..., 1::8] = h - obboxes[..., 1::8] - 1
        flipped[..., 3::8] = h - obboxes[..., 3::8] - 1
        flipped[..., 5::8] = h - obboxes[..., 5::8] - 1
        flipped[..., 7::8] = h - obboxes[..., 7::8] - 1
    else:
        raise ValueError(
            'Invalid flipping direction "{}"'.format(direction))
    return flipped

def obbox_mapping_back(obboxes, img_shape, scale_factor, flip, flip_direction):
    """ mapping back the coordinates of oriented bounding boxes(poly).
    """
    fliped_obboxes = obbox_flip(obboxes, img_shape, flip_direction) if flip else obboxes
    fliped_obboxes = fliped_obboxes / scale_factor
    return fliped_obboxes


def merge_aug_poly_results(aug_obboxes, aug_scores, img_metas):
    """
    Merge augmented detection obboxes and scores.

    Args:
        aug_obboxes (list[Tensor]) : shape (n, 8*#class)
        aug_scores (list[Tensor]): shape (n, #class)

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_obboxes = []
    for obboxes, img_info in zip(aug_obboxes, img_metas):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']
        flip = img_info[0]['flip']
        flip_direction = img_info[0]['flip_direction']
        obboxes = obbox_mapping_back(obboxes, img_shape, scale_factor, flip, flip_direction)
        recovered_obboxes.append(obboxes.float())

    obboxes = torch.cat(recovered_obboxes, dim=0)
    if aug_scores is None:
        return obboxes
    else:
        scores = torch.cat(aug_scores, dim=0)
        return obboxes, scores









