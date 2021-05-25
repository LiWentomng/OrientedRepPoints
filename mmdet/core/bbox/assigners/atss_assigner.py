import torch


from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from mmdet.ops.iou import convex_iou
from mmdet.ops.point_justify import pointsJf
from mmdet.ops.minareabbox import find_minarea_rbbox


class ATSSAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self, topk):
        self.topk = topk

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as postive
        6. limit the positive sample's center in gt


        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000

        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt
        # print('######')
        # print('gt_bboxes', gt_bboxes.shape)
        # print('bboxes', bboxes.shape)
        # convert_box = find_minarea_rbbox(bboxes)
        # print('convert_box', convert_box.shape)
        # gt_corners_point = self.corners_to_edge_points_10(gt_bboxes, 20)
        # convert_box_corners_point = self.corners_to_edge_points_10(convert_box, 20)
        # print('gt_corners_point', gt_corners_point.shape)
        # print('convert_box_corners_point', convert_box_corners_point.shape)

        overlaps = self.convex_overlaps(gt_bboxes, bboxes)# (gt_bboxes, points)
        # print('overlaps', overlaps.shape)

        # gt_bboxes_distance = gt_bboxes.unsqueeze(dim=0).repeat(num_bboxes, 1, 1).reshape(-1, 8)
        # bboxes_distance = convert_box.unsqueeze(dim=1).repeat(1, num_gt, 1).reshape(-1, 8)
        # print('gt_bboxes_distance', gt_bboxes_distance.shape)
        # print('bboxes_distance', bboxes_distance.shape)
        # gt_corners_point_distance = self.corners_to_edge_points_10(gt_bboxes_distance, 20)
        # box_corners_point_distance = self.corners_to_edge_points_10(bboxes_distance, 20)

        # gt_bboxes_distance = gt_bboxes.unsqueeze(dim=0).repeat(num_bboxes, 1, 1).reshape(-1, 8)
        # bboxes_distance = bboxes.unsqueeze(dim=1).repeat(1, num_gt, 1).reshape(-1, 18)
        # print('gt_bboxes_distance', gt_bboxes_distance.shape)
        # print('bboxes_distance', bboxes_distance.shape)

        # gt_boxes_x = gt_bboxes_distance[:, 0::2].unsqueeze(dim=2)
        # gt_boxes_y = gt_bboxes_distance[:, 1::2].unsqueeze(dim=2)
        # gt_boxes_distance = torch.cat([gt_boxes_x, gt_boxes_y], dim=2)
        # print('gt_boxes_distance', gt_boxes_distance.shape)
        # #
        # pred_point_x = bboxes_distance[:, 0::2].unsqueeze(dim=2)
        # pred_point_y = bboxes_distance[:, 1::2].unsqueeze(dim=2)
        # pred_point_distance = torch.cat([pred_point_x, pred_point_y], dim=2)
        # print('pred_point_distance', pred_point_distance.shape)

        # giou_overlaps = self.convex_giou_overlaps(gt_bboxes, bboxes)
        # print('giou_overlaps', giou_overlaps.shape)

        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_zeros((num_bboxes, ),
                                                     dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        # the center of poly
        gt_bboxes_hbb = self.get_horizontal_bboxes(gt_bboxes)  #convert to hbb

        gt_cx = (gt_bboxes_hbb[:, 0] + gt_bboxes_hbb[:, 2]) / 2.0
        gt_cy = (gt_bboxes_hbb[:, 1] + gt_bboxes_hbb[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        #calculat center of points
        #y_first True or False?

        '''
        (1)points to rbox 
        (2)calculate the center of rbox
        # print('predicted_points', bboxes.shape)
        rboxes = find_minarea_rbbox(bboxes)
        # print('rboxes', rboxes.shape)
        rbboxes_cx = (rboxes[:, 0] + rboxes[:, 4]) / 2.0
        rbboxes_cy = (rboxes[:, 1] + rboxes[:, 5]) / 2.0
        bboxes_points = torch.stack((rbboxes_cx, rbboxes_cy), dim=1)
        # rbboxes_cx_1 = (rboxes[:, 2] + rboxes[:, 6]) / 2.0
        # rbboxes_cy_1 = (rboxes[:, 3] + rboxes[:, 7]) / 2.0
        # print(torch.abs(rbboxes_cx - pts_x_mean).sum())
        # print(torch.abs(rbboxes_cy - pts_y_mean).sum())
        '''

        bboxes = bboxes.reshape(-1, 9, 2)
        # y_first False
        pts_x = bboxes[:, :, 0::2]  #
        pts_y = bboxes[:, :, 1::2]  #

        pts_x_mean = pts_x.mean(dim=1).squeeze()
        pts_y_mean = pts_y.mean(dim=1).squeeze()
        # print('pts_x_mean', pts_x_mean.shape, pts_x_mean)
        # print('pts_y_mean', pts_y_mean.shape, pts_y_mean)

        # bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        # bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((pts_x_mean, pts_y_mean), dim=1)


        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # Selecting candidates based on the center distance
        candidate_idxs = []
        _, topk_idxs = distances.topk(self.topk, dim=0, largest=False)
        candidate_idxs.append(topk_idxs)
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # start_idx = 0
        # for level, bboxes_per_level in enumerate(num_level_bboxes):
        #     # on each pyramid level, for each gt,
        #     # select k bbox whose center are closest to the gt center
        #     end_idx = start_idx + bboxes_per_level
        #     distances_per_level = distances[start_idx:end_idx, :]
        #     _, topk_idxs_per_level = distances_per_level.topk(
        #         self.topk, dim=0, largest=False)
        #     candidate_idxs.append(topk_idxs_per_level + start_idx)
        #     start_idx = end_idx
        # candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        # clamp neg min threshold
        # overlaps_thr_per_gt = overlaps_thr_per_gt.clamp_min(0.3)

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]
        # print('is_pos', is_pos.shape)

        # limit the positive sample's center in gt


        inside_flag = torch.full([num_bboxes, num_gt], 0.).to(gt_bboxes.device).float()
        # print('inside_flag', inside_flag.shape, inside_flag.dtype)
        # print('gt_bboxes', gt_bboxes.shape, gt_bboxes.dtype)
        pointsJf(bboxes_points, \
                 gt_bboxes,\
                inside_flag)
        # print('inside_flag', inside_flag, torch.where(inside_flag>0))
        is_in_gts = inside_flag[candidate_idxs, torch.arange(num_gt)].to(is_pos.dtype)


        '''
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes

        ep_bboxes_cx = pts_x_mean.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = pts_y_mean.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)
        # print('candidate_idxs', candidate_idxs.shape, candidate_idxs)

        # calculate the left, top, right, bottom distance between positive bbox center and gt side
        # print('ep_bboxes_cx[candidate_idxs].view(-1, num_gt)', ep_bboxes_cx[candidate_idxs].view(-1, num_gt).shape, ep_bboxes_cx[candidate_idxs].view(-1, num_gt))
        # print('ep_bboxes_cy[candidate_idxs].view(-1, num_gt)', ep_bboxes_cy[candidate_idxs].view(-1, num_gt).shape, ep_bboxes_cx[candidate_idxs].view(-1, num_gt))

        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes_hbb[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes_hbb[:, 1]
        r_ = gt_bboxes_hbb[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes_hbb[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        # print('l_', l_.shape) #[45, 2] 2代表有几个gt box
        # print('gt_bboxes', gt_bboxes.shape)# [2, 4]
        # print()
        # print('torch.stack([l_, t_, r_, b_], dim=1)', torch.stack([l_, t_, r_, b_], dim=1).shape)#在第1维升了一维，进行合并
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01 #最小值大于0.01,也就是全部大于0.01,也就是中心点在框里面。
        # print('is_in_gts', is_in_gts.shape) # [45, 2]
        '''

        is_pos = is_pos & is_in_gts

        # print('is_pos', is_pos)
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        candidate_idxs = candidate_idxs.view(-1)

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)


    def convex_overlaps(self, gt_rbboxes, points):
        overlaps = convex_iou(points, gt_rbboxes)
        # overlaps = overlaps.transpose(1, 0)
        return overlaps

    # def convex_giou_overlaps(self, gt_rbboxes, points):
    #     giou_overlaps, giou_overlaps_grads = convex_giou(points, gt_rbboxes)
    #     return giou_overlaps


    def get_horizontal_bboxes(self, gt_rbboxes):
        gt_xs, gt_ys = gt_rbboxes[:, 0::2], gt_rbboxes[:, 1::2]
        gt_xmin, _ = gt_xs.min(1)
        gt_ymin, _ = gt_ys.min(1)
        gt_xmax, _ = gt_xs.max(1)
        gt_ymax, _ = gt_ys.max(1)
        gt_rect_bboxes = torch.cat([gt_xmin[:, None], gt_ymin[:, None],
                                    gt_xmax[:, None], gt_ymax[:, None]], dim=1)
        return gt_rect_bboxes


    def corners_to_edge_points_10(self, corners, points_num):
        device = corners.device
        corners_xs, corners_ys = corners[:, 0::2], corners[:, 1::2]
        # print('corners_xs_10', corners_xs.shape, corners_xs)
        # print('corners_ys_10', corners_ys.shape, corners_ys)

        first_edge_x_points = corners_xs[:, 0:2]
        first_edge_y_points = corners_ys[:, 0:2]

        sec_edge_x_points = corners_xs[:, 1:3]
        sec_edge_y_points = corners_ys[:, 1:3]

        third_edge_x_points = corners_xs[:, 2:4]
        third_edge_y_points = corners_ys[:, 2:4]

        four_edge_x_points_s = corners_xs[:, 3]
        four_edge_y_points_s = corners_ys[:, 3]
        four_edge_x_points_e = corners_xs[:, 0]
        four_edge_y_points_e = corners_ys[:, 0]

        edge_ratio = torch.linspace(0, 1, points_num).to(device).repeat(corners.shape[0], 1)

        all_1_edge_x_points = edge_ratio *first_edge_x_points[:, 1:2] + \
                              (1- edge_ratio)* first_edge_x_points[:, 0:1]
        all_1_edge_y_points =  edge_ratio * first_edge_y_points[:, 1:2] + \
                                 (1- edge_ratio)* first_edge_y_points[:, 0:1]

        # print('all_1_edge_x_points_10', all_1_edge_x_points.shape, all_1_edge_x_points)
        # print('all_1_edge_y_points_10', all_1_edge_y_points.shape, all_1_edge_y_points)

        all_2_edge_x_points = edge_ratio *sec_edge_x_points[:, 1:2]  + \
                                 (1- edge_ratio)* sec_edge_x_points[:, 0:1]
        all_2_edge_y_points = edge_ratio * sec_edge_y_points[:, 1:2] + \
                                 (1- edge_ratio)* sec_edge_y_points[:, 0:1]

        # print('all_2_edge_x_points_10', all_2_edge_x_points.shape, all_2_edge_x_points)
        # print('all_2_edge_y_points_10', all_2_edge_y_points.shape, all_2_edge_y_points)

        all_3_edge_x_points = edge_ratio *third_edge_x_points[:, 1:2]  + \
                                 (1- edge_ratio)* third_edge_x_points[:, 0:1]
        all_3_edge_y_points = edge_ratio * third_edge_y_points[:, 1:2] + \
                                 (1- edge_ratio)* third_edge_y_points[:, 0:1]

        all_4_edge_x_points = edge_ratio *four_edge_x_points_e.unsqueeze(1)  + \
                                 (1- edge_ratio)* four_edge_x_points_s.unsqueeze(1)
        all_4_edge_y_points = edge_ratio * four_edge_y_points_e.unsqueeze(1) + \
                                 (1- edge_ratio)* four_edge_y_points_s.unsqueeze(1)

        # print('all_4_edge_x_points_10', all_4_edge_x_points.shape, all_4_edge_x_points)
        # print('all_4_edge_y_points_10', all_4_edge_y_points.shape, all_4_edge_y_points)

        # torch.Size([b, 40])
        all_x_points = torch.cat([all_1_edge_x_points, all_2_edge_x_points,
                                  all_3_edge_x_points, all_4_edge_x_points], dim=1).unsqueeze(dim=2)
        # print('all_x_points', all_x_points.shape, all_x_points)

        all_y_points = torch.cat([all_1_edge_y_points, all_2_edge_y_points,
                                  all_3_edge_y_points, all_4_edge_y_points], dim=1).unsqueeze(dim=2)
        # print('all_y_points', all_y_points.shape, all_y_points)

        all_points = torch.cat([all_x_points, all_y_points], dim=2)
        # print('all_points_10', all_points.shape, all_points)
        return all_points