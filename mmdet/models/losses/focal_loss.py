import torch.nn as nn
import torch.nn.functional as F
import torch

from mmdet.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from ..registry import LOSSES
from .utils import weight_reduce_loss

def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred, target, gamma, alpha)
    # TODO: find a proper way to handle the shape of weight
    if weight is not None:
        weight = weight.view(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

def separate_sigmoid_focal_loss(pred,
                                target,
                                weight=None,
                                gamma=2.0,
                                alpha=0.25,
                                reduction='mean',
                                avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)

    pos_inds = target.eq(1)
    neg_inds = target.lt(1)

    pos_weights = weight[pos_inds]

    pos_pred = pred_sigmoid[pos_inds]
    neg_pred = pred_sigmoid[neg_inds]

    pos_loss = -torch.log(pos_pred) * torch.pow(1 - pos_pred, gamma) * pos_weights * alpha
    neg_loss = -torch.log(1 - neg_pred) * torch.pow(neg_pred, gamma) * (1 - alpha)

    if pos_pred.nelement() == 0:
        loss = neg_loss.sum() / avg_factor
    else:
        loss = pos_loss.sum() / pos_weights.sum() + neg_loss.sum() / avg_factor

    return loss

@LOSSES.register_module
class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls
    
@LOSSES.register_module()
class SEPFocalLoss(nn.Module):

    def __init__(self,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(SEPFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * separate_sigmoid_focal_loss(
            pred,
            target,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss_cls
    
def balance_sigmoid(pred, num_instances):
    class_num = pred.size(-1)
    all_instance = num_instances.sum()
    ave_instance = all_instance / class_num
    balanced_factor = torch.log(ave_instance / num_instances * \
                                 (all_instance - num_instances) / (all_instance - ave_instance)
                                ).reshape(1, -1).type_as(pred)
    
    pred_score = pred - balanced_factor
    pred_sigmoid = pred_score.exp() / (1 + pred_score.exp())
    
    return pred_sigmoid

def my_py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          num_instances=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    num_classes = pred.size(-1)
    device = pred.device
    class_range = torch.arange(1, num_classes + 1, device=device).type_as(pred).unsqueeze(0)

    if num_instances is None:
        pred_sigmoid = pred.sigmoid()
    else:
        pred_sigmoid = balance_sigmoid(pred, num_instances)
        
    target = target.type_as(pred).unsqueeze(1)
    
    term1 = (1 - pred_sigmoid) ** gamma * pred_sigmoid.log()
    term2 = pred_sigmoid ** gamma * (1 - pred_sigmoid).log()
    
    loss = -(target == class_range).float() * term1 * alpha - \
            ((target != class_range) * (target >= 0)).float() * term2 * (1 - alpha)

    loss = loss.sum() / avg_factor
    return loss

@LOSSES.register_module
class PYFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(PYFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                num_instances=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * my_py_sigmoid_focal_loss(
                pred,
                target,
                weight,
                num_instances=num_instances,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls