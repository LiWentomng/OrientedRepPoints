import torch
import torch.nn as nn

from mmdet.ops.chamfer_2d import Chamfer2D

def ChamferDistance2D(point_set_1, point_set_2, distance_weight = 0.05, eps = 1e-12, use_cuda= True):

    chamfer = Chamfer2D() #先把类赋给一个变量  # 再直接调用这个变量即调用使用forward
    assert point_set_1.dim() == point_set_2.dim()
    assert point_set_1.shape[-1] == point_set_2.shape[-1]
    if point_set_1.dim() <= 3:
        if use_cuda:
            # chamfer loss
            dist1, dist2, _, _ = chamfer(point_set_1, point_set_2)
            dist1 = torch.sqrt(torch.clamp(dist1, eps))
            dist2 = torch.sqrt(torch.clamp(dist2, eps))
            dist = (dist1.mean(-1) + dist2.mean(-1)) / 2.0

        else:
            dist = chamfer(point_set_1, point_set_2)
        return dist * distance_weight

