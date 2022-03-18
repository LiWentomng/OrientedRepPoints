import torch
import torch.nn as nn

from ..registry import LOSSES
from mmdet.ops.point_justify import pointsJf

@LOSSES.register_module
class SpatialBorderLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(SpatialBorderLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pts, gt_bboxes, weight, y_first=False, *args, **kwargs):
        loss = self.loss_weight * weighted_spatial_border_loss(
            pts, gt_bboxes, weight, y_first=y_first, *args, **kwargs)
        return loss

def spatial_border_loss(pts, gt_bboxes, reduction='mean', y_first=False):

    num_gt, num_pts = gt_bboxes.size(0), pts.size(0)
    loss = pts.new_zeros([0])

    if num_gt > 0:
        inside_flag_1 = torch.full([num_pts, num_gt], 0.).to(gt_bboxes.device).float()
        pt_1 = pts[:, 0:2].reshape(num_pts, 2).contiguous()
        pointsJf(pt_1, gt_bboxes, inside_flag_1)
        inside_flag_1 = torch.diag(inside_flag_1)

        inside_flag_2 = torch.full([num_pts, num_gt], 0.).to(gt_bboxes.device).float()
        pt_2 = pts[:, 2:4].reshape(num_pts, 2).contiguous()
        pointsJf(pt_2, gt_bboxes, inside_flag_2)
        inside_flag_2 = torch.diag(inside_flag_2)

        inside_flag_3 = torch.full([num_pts, num_gt], 0.).to(gt_bboxes.device).float()
        pt_3 = pts[:, 4:6].reshape(num_pts, 2).contiguous()
        pointsJf(pt_3, gt_bboxes, inside_flag_3)
        inside_flag_3 = torch.diag(inside_flag_3)

        inside_flag_4 = torch.full([num_pts, num_gt], 0.).to(gt_bboxes.device).float()
        pt_4 = pts[:, 6:8].reshape(num_pts, 2).contiguous()
        pointsJf(pt_4, gt_bboxes, inside_flag_4)
        inside_flag_4 = torch.diag(inside_flag_4)

        inside_flag_5 = torch.full([num_pts, num_gt], 0.).to(gt_bboxes.device).float()
        pt_5 = pts[:, 8:10].reshape(num_pts, 2).contiguous()
        pointsJf(pt_5, gt_bboxes, inside_flag_5)
        inside_flag_5 = torch.diag(inside_flag_5)

        inside_flag_6 = torch.full([num_pts, num_gt], 0.).to(gt_bboxes.device).float()
        pt_6 = pts[:, 10:12].reshape(num_pts, 2).contiguous()
        pointsJf(pt_6, gt_bboxes, inside_flag_6)
        inside_flag_6 = torch.diag(inside_flag_6)

        inside_flag_7 = torch.full([num_pts, num_gt], 0.).to(gt_bboxes.device).float()
        pt_7 = pts[:, 12:14].reshape(num_pts, 2).contiguous()
        pointsJf(pt_7, gt_bboxes, inside_flag_7)
        inside_flag_7 = torch.diag(inside_flag_7)

        inside_flag_8 = torch.full([num_pts, num_gt], 0.).to(gt_bboxes.device).float()
        pt_8 = pts[:, 14:16].reshape(num_pts, 2).contiguous()
        pointsJf(pt_8, gt_bboxes, inside_flag_8)
        inside_flag_8 = torch.diag(inside_flag_8)

        inside_flag_9 = torch.full([num_pts, num_gt], 0.).to(gt_bboxes.device).float()
        pt_9 = pts[:, 16:18].reshape(num_pts, 2).contiguous()
        pointsJf(pt_9, gt_bboxes, inside_flag_9)
        inside_flag_9 = torch.diag(inside_flag_9)

        inside_flag = torch.stack([inside_flag_1, inside_flag_2, inside_flag_3, inside_flag_4, inside_flag_5,
                                 inside_flag_6, inside_flag_7, inside_flag_8, inside_flag_9], dim=1)

        pts = pts.reshape(-1, 9, 2)
        out_border_pts = pts[torch.where(inside_flag == 0)]

        if out_border_pts.size(0) > 0:
            corres_gt_boxes = gt_bboxes[torch.where(inside_flag == 0)[0]]
            corres_gt_boxes_center_x = (corres_gt_boxes[:, 0] + corres_gt_boxes[:, 4]) / 2.0
            corres_gt_boxes_center_y = (corres_gt_boxes[:, 1] + corres_gt_boxes[:, 5]) / 2.0
            corres_gt_boxes_center = torch.stack([corres_gt_boxes_center_x, corres_gt_boxes_center_y], dim=1)
            distance_out_pts = 0.2*(((out_border_pts - corres_gt_boxes_center)**2).sum(dim=1).sqrt())
            loss = distance_out_pts.sum() / out_border_pts.size(0)

    return loss

def weighted_spatial_border_loss(pts, gt_bboxes, weight, avg_factor=None, y_first=False):

    weight = weight.unsqueeze(dim=1).repeat(1, 4)
    assert weight.dim() == 2
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = spatial_border_loss(pts, gt_bboxes, y_first=y_first, reduction='none')  # (n, 4, num_points)
    return torch.sum(loss)[None] / avg_factor

