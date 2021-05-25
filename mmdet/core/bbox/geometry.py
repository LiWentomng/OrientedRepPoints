import torch
import numpy as np
# from .bbox import bbox_overlaps_cython
import DOTA_devkit.polyiou as polyiou
# from shapely.geometry import Polygon
# import shapely

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4) in <x1, y1, x2, y2> format.
        bboxes2 (Tensor): shape (n, 4) in <x1, y1, x2, y2> format.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps(bboxes1, bboxes2)
        tensor([[0.5238, 0.0500, 0.0041],
                [0.0323, 0.0452, 1.0000],
                [0.0000, 0.0000, 0.0000]])

    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 0, 10, 9],
        >>> ])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious


# def rbbox_overlaps_cy_warp(rbboxes, query_boxes):
#     # TODO: first calculate the hbb overlaps, for overlaps > 0, calculate the obb overlaps
#     # import pdb
#     # pdb.set_trace()
#     # print('rbboxes', rbboxes.shape[0])
#     # print('query_boxes', query_boxes.shape)
#     box_device = query_boxes.device
#     # polys_np
#     rbboxes_np = rbboxes.cpu().detach().numpy().astype(np.float)
#     # print('rbboxes_np.shape', rbboxes_np.shape)
#     #query_polys_np
#     query_boxes_np = query_boxes.cpu().detach().numpy().astype(np.float)
#
#
#     # polys_np = RotBox2Polys(boxes_np)
#     # TODO: change it to only use pos gt_masks
#     # polys_np = mask2poly(gt_masks)
#     # polys_np = np.array(Tuplelist2Polylist(polys_np)).astype(np.float)
#
#     # polys_np = RotBox2Polys(rbboxes).astype(np.float)
#     # query_polys_np = RotBox2Polys(query_boxes_np)
#
#     h_bboxes_np = poly2bbox(rbboxes_np)
#     h_query_bboxes_np = poly2bbox(query_boxes_np)
#
#     # hious
#     ious = bbox_overlaps_cython(h_bboxes_np, h_query_bboxes_np)
#     # ious_align = np.zeros(shape=(rbboxes_np.shape[0]))
#     import pdb
#     # pdb.set_trace()
#     inds = np.where(ious > 0)
#     for index in range(len(inds[0])):
#         box_index = inds[0][index]
#         query_box_index = inds[1][index]
#
#         box = rbboxes_np[box_index]
#         query_box = query_boxes_np[query_box_index]
#
#         # calculate obb iou
#         # import pdb
#         # pdb.set_trace()
#         overlap = polyiou.iou_poly(polyiou.VectorDouble(box), polyiou.VectorDouble(query_box))
#         ious[box_index][query_box_index] = overlap
#         # print('ious', ious.shape)
#         # if box_index == query_box_index:
#         #     ious_align[box_index] = overlap
#             # print('ious_align', ious_align.shape)
#
#     return torch.from_numpy(ious).to(box_device)
#     # return torch.from_numpy(ious).to(box_device), torch.from_numpy(ious_align).to(box_device)


def poly2bbox(polys):
    """
    without label
    :param polys: (x1, y1, ..., x4, y4) (n, 8)
    :return: boxes: (xmin, ymin, xmax, ymax) (n, 4)
    """
    n = polys.shape[0]
    xs = np.reshape(polys, (n, 4, 2))[:, :, 0]
    ys = np.reshape(polys, (n, 4, 2))[:, :, 1]

    xmin = np.min(xs, axis=1)
    ymin = np.min(ys, axis=1)
    xmax = np.max(xs, axis=1)
    ymax = np.max(ys, axis=1)

    xmin = xmin[:, np.newaxis]
    ymin = ymin[:, np.newaxis]
    xmax = xmax[:, np.newaxis]
    ymax = ymax[:, np.newaxis]

    return np.concatenate((xmin, ymin, xmax, ymax), 1)