import torch
from mmdet.ops.nms import nms_wrapper

def multiclass_obbnms(multi_oboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   multi_oreppoints=None):
    """
    OBNMS for multi-class oriented boxes.

    Args:
        multi_oboxes (Tensor): shape (n, #class*8) or (n, 8)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, oboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num oboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS.
        multi_oreppoints: oriented reppoints.

    Returns:
        tuple: (oboxes, labels), tensors of shape (k, 8) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_oboxes.shape[1] > 8:
        oboxes = multi_oboxes.view(multi_scores.size(0), -1, 8)[:, 1:]
    else:
        oboxes = multi_oboxes[:, None].expand(-1, num_classes, 8)
        
    if multi_oreppoints is not None:
        oreppoints = multi_oreppoints[:, None].expand(-1, num_classes, multi_oreppoints.size(-1))
        
    scores = multi_scores[:, 1:]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    oboxes = oboxes[valid_mask]
    
    if multi_oreppoints is not None:
        oreppoints = oreppoints[valid_mask]
        
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]
    
    if oboxes.numel() == 0:
        if multi_oreppoints is None:
            oboxes = multi_oboxes.new_zeros((0, 9))
        else:
            oboxes = multi_oboxes.new_zeros((0, oreppoints.size(-1) + 9))
        labels = multi_oboxes.new_zeros((0, ), dtype=torch.long)
        return oboxes, labels

    max_coordinate = oboxes.max()
    offsets = labels.to(oboxes) * (max_coordinate + 1)
    oboxes_for_nms = oboxes + offsets[:, None]
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'rnms')
    nms_op = getattr(nms_wrapper, nms_type)
    
    dets, keep = nms_op(
        torch.cat([oboxes_for_nms, scores[:, None]], 1), **nms_cfg_)
    
    oboxes = oboxes[keep]
    if multi_oreppoints is not None:
        oreppoints = oreppoints[keep]
        oboxes = torch.cat([oreppoints, oboxes], dim=1)

    scores = dets[:, -1]  # soft_nms will modify scores
    labels = labels[keep]

    if keep.size(0) > max_num:
        _, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        oboxes = oboxes[inds]
        scores = scores[inds]
        labels = labels[inds]

    return torch.cat([oboxes, scores[:, None]], 1), labels
