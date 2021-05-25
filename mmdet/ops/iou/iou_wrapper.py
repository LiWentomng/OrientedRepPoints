import numpy as np
import torch

from . import convex_giou_cuda
from . import convex_iou_cuda

'''
adapt convex hull of points set for iou/giou calculation 
from SDL-GuoZonghao: https://github.com/SDL-GuoZonghao/BeyondBoundingBox/blob/main/mmdet/ops/iou/iou_wrapper.py
'''

def convex_giou(pred, target):
    convex_giou_grad = convex_giou_cuda.convex_giou(pred, target)
    convex_giou_grad = convex_giou_grad.reshape(-1, 19)
    convex_giou = convex_giou_grad[:, -1]
    points_grad = convex_giou_grad[:, 0:-1]
    return convex_giou, points_grad

def convex_iou(pred, target):
    ex_num, gt_num = pred.size(0), target.size(0)
    convex_ious = convex_iou_cuda.convex_iou(pred, target)
    convex_ious = convex_ious.reshape(ex_num, gt_num)
    return convex_ious

def convex_overlaps(gt_rbboxes, points):
    overlaps = convex_iou(points, gt_rbboxes)
    overlaps = overlaps.transpose(1, 0)
    return overlaps