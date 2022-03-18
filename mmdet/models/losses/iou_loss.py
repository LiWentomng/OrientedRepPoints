import torch
import torch.nn as nn

from mmdet.core import bbox_overlaps
from ..registry import LOSSES
from .utils import weighted_loss

from torch.autograd import Function
from torch.autograd.function import once_differentiable
from mmdet.ops.iou import convex_giou

@weighted_loss
def iou_loss(pred, target, eps=1e-6):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    loss = -ious.log()
    return loss

@LOSSES.register_module
class IoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(IoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * iou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


'''
gt box and  convex hull of points set for giou loss
from SDL-GuoZonghao:https://github.com/SDL-GuoZonghao/BeyondBoundingBox/blob/main/mmdet/models/losses/iou_loss.py
'''

class GIoULossFuction(Function):
    @staticmethod
    def forward(ctx, pred, target, weight=None, reduction=None, avg_factor=None, loss_weight=1.0):
        ctx.save_for_backward(pred)

        convex_gious, grad = convex_giou(pred, target)
        loss = 1 - convex_gious
        if weight is not None:
            loss = loss * weight
            grad = grad * weight.reshape(-1, 1)
        if reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'mean':
            loss = loss.mean()
        else:
            loss = loss

        # _unvalid_grad_filter
        eps = 1e-6
        unvaild_inds = torch.nonzero((grad > 1).sum(1))[:, 0]
        grad[unvaild_inds] = eps

        # _reduce_grad
        reduce_grad = -grad / grad.size(0) * loss_weight
        ctx.convex_points_grad = reduce_grad
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, input=None):
        convex_points_grad = ctx.convex_points_grad
        return convex_points_grad, None, None, None, None, None

convex_giou_loss = GIoULossFuction.apply
@LOSSES.register_module
class GIoULoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight.unsqueeze(-1)).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * convex_giou_loss(
            pred,
            target,
            weight,
            reduction,
            avg_factor,
            self.loss_weight)
        return loss

