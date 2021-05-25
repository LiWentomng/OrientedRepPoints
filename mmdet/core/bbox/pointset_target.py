import torch
from ..utils import multi_apply, unmap
from .samplers import PseudoSampler
from .assign_sampling import assign_and_sample, build_assigner

def init_pointset_target(proposals_list,
                 valid_flag_list,
                 gt_rbboxes_list,
                 img_metas,
                 cfg,
                 gt_rbboxes_ignore_list=None,
                 gt_labels_list=None,
                 label_channels=1,
                 sampling=True,
                 unmap_outputs=True):
    num_imgs = len(img_metas)
    assert len(proposals_list) == len(valid_flag_list) == num_imgs
    # points number of multi levels
    num_level_proposals = [points.size(0) for points in proposals_list[0]]
    # concat all level points and flags to a single tensor
    for i in range(num_imgs):
        assert len(proposals_list[i]) == len(valid_flag_list[i])
        proposals_list[i] = torch.cat(proposals_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])

    # compute targets for each image
    if gt_rbboxes_ignore_list is None:
        gt_rbboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    (all_labels, all_label_weights, all_rbbox_gt, all_proposals,
     all_proposal_weights, pos_inds_list, neg_inds_list, all_gt_inds_list) = multi_apply(
        init_pointset_target_single,
        proposals_list,
        valid_flag_list,
        gt_rbboxes_list,
        gt_rbboxes_ignore_list,
        gt_labels_list,
        cfg=cfg,
        label_channels=label_channels,
        sampling=sampling,
        unmap_outputs=unmap_outputs)
    # no valid points
    if any([labels is None for labels in all_labels]):
        return None

    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    labels_list = images_to_levels(all_labels, num_level_proposals)
    label_weights_list = images_to_levels(all_label_weights,
                                          num_level_proposals)
    rbbox_gt_list = images_to_levels(all_rbbox_gt, num_level_proposals)
    proposals_list = images_to_levels(all_proposals, num_level_proposals)
    proposal_weights_list = images_to_levels(all_proposal_weights,
                                             num_level_proposals)
    gt_inds_list = images_to_levels(all_gt_inds_list, num_level_proposals)
    return (labels_list, label_weights_list, rbbox_gt_list, proposals_list,
            proposal_weights_list, num_total_pos, num_total_neg, gt_inds_list)


def init_pointset_target_single(flat_proposals,
                        valid_flags,
                        gt_rbboxes,
                        gt_rbboxes_ignore,
                        gt_labels,
                        cfg,
                        label_channels=1,
                        sampling=True,
                        unmap_outputs=True):
    inside_flags = valid_flags
    if not inside_flags.any():
        return (None,) * 7
    # assign gt and sample proposals
    proposals = flat_proposals[inside_flags, :]
    if sampling:
        assign_result, sampling_result = assign_and_sample(
            proposals, gt_rbboxes, gt_rbboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(proposals, gt_rbboxes, gt_rbboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, proposals,
                                              gt_rbboxes)
    gt_inds = assign_result.gt_inds
    num_valid_proposals = proposals.shape[0]
    rbbox_gt = proposals.new_zeros([num_valid_proposals, 8])
    pos_proposals = torch.zeros_like(proposals)
    proposals_weights = proposals.new_zeros(num_valid_proposals)
    labels = proposals.new_zeros(num_valid_proposals, dtype=torch.long)
    label_weights = proposals.new_zeros(num_valid_proposals, dtype=torch.float)
    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_gt_rbboxes = sampling_result.pos_gt_rbboxes
        rbbox_gt[pos_inds, :] = pos_gt_rbboxes
        pos_proposals[pos_inds, :] = proposals[pos_inds, :]
        proposals_weights[pos_inds] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of proposals
    if unmap_outputs:
        num_total_proposals = flat_proposals.size(0)
        labels = unmap(labels, num_total_proposals, inside_flags)
        label_weights = unmap(label_weights, num_total_proposals, inside_flags)
        rbbox_gt = unmap(rbbox_gt, num_total_proposals, inside_flags)
        pos_proposals = unmap(pos_proposals, num_total_proposals, inside_flags)
        proposals_weights = unmap(proposals_weights, num_total_proposals,
                                  inside_flags)
        gt_inds = unmap(gt_inds, num_total_proposals, inside_flags)

    return (labels, label_weights, rbbox_gt, pos_proposals, proposals_weights,
            pos_inds, neg_inds, gt_inds)


def refine_pointset_target(proposals_list,
                valid_flag_list,
                gt_rbboxes_list,
                img_metas,
                cfg,
                gt_rbboxes_ignore_list=None,
                gt_labels_list=None,
                label_channels=1,
                sampling=True,
                unmap_outputs=True):
    num_imgs = len(img_metas)
    assert len(proposals_list) == len(valid_flag_list) == num_imgs
    # concat all level points and flags to a single tensor
    for i in range(num_imgs):
        assert len(proposals_list[i]) == len(valid_flag_list[i])
        proposals_list[i] = torch.cat(proposals_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])

    # compute targets for each image
    if gt_rbboxes_ignore_list is None:
        gt_rbboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    (all_labels, all_label_weights, all_rbbox_gt, all_proposals,
     all_proposal_weights, pos_inds_list, neg_inds_list, all_gt_inds) = multi_apply(
        refine_pointset_target_single,
        proposals_list,
        valid_flag_list,
        gt_rbboxes_list,
        gt_rbboxes_ignore_list,
        gt_labels_list,
        cfg=cfg,
        label_channels=label_channels,
        sampling=sampling,
        unmap_outputs=unmap_outputs)

    pos_inds = []
    pos_gt_index = []
    for i, single_labels in enumerate(all_labels):
        pos_mask = single_labels > 0
        pos_inds.append(pos_mask.nonzero().view(-1))
        pos_gt_index.append(all_gt_inds[i][pos_mask.nonzero().view(-1)])

    return (all_labels, all_label_weights, all_rbbox_gt, all_proposals,
            all_proposal_weights, pos_inds, pos_gt_index)


def refine_pointset_target_single(flat_proposals,
                        valid_flags,
                        gt_rbboxes,
                        gt_rbboxes_ignore,
                        gt_labels,
                        cfg,
                        label_channels=1,
                        sampling=True,
                        unmap_outputs=True):
    inside_flags = valid_flags
    if not inside_flags.any():
        return (None,) * 7
    # assign gt and sample proposals
    proposals = flat_proposals[inside_flags, :]
    if sampling:
        assign_result, sampling_result = assign_and_sample(
            proposals, gt_rbboxes, gt_rbboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(proposals, gt_rbboxes,
                                             gt_rbboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, proposals,
                                              gt_rbboxes)
    gt_inds = assign_result.gt_inds
    num_valid_proposals = proposals.shape[0]
    rbbox_gt = proposals.new_zeros([num_valid_proposals, 8])
    pos_proposals = torch.zeros_like(proposals)
    proposals_weights = proposals.new_zeros(num_valid_proposals)
    labels = proposals.new_zeros(num_valid_proposals, dtype=torch.long)
    label_weights = proposals.new_zeros(num_valid_proposals, dtype=torch.float)
    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_gt_rbboxes = sampling_result.pos_gt_rbboxes
        rbbox_gt[pos_inds, :] = pos_gt_rbboxes
        pos_proposals[pos_inds, :] = proposals[pos_inds, :]
        proposals_weights[pos_inds] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0
    # map up to original set of proposals
    if unmap_outputs:
        num_total_proposals = flat_proposals.size(0)
        labels = unmap(labels, num_total_proposals, inside_flags)
        label_weights = unmap(label_weights, num_total_proposals, inside_flags)
        rbbox_gt = unmap(rbbox_gt, num_total_proposals, inside_flags)
        pos_proposals = unmap(pos_proposals, num_total_proposals, inside_flags)
        proposals_weights = unmap(proposals_weights, num_total_proposals,
                                  inside_flags)
        gt_inds = unmap(gt_inds, num_total_proposals, inside_flags)
    return (labels, label_weights, rbbox_gt, pos_proposals, proposals_weights,
            pos_inds, neg_inds, gt_inds)


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets

def levels_to_images(mlvl_tensor, flatten=False):

    """
    convert targets by levels to targets by feature level.
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    if flatten:
        channels = mlvl_tensor[0].size(-1)
    else:
        channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        if not flatten:
            t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]