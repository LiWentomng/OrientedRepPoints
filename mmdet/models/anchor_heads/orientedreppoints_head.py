from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.cnn import constant_init
from mmdet.core import (PointGenerator, multi_apply, multiclass_rnms,
                       levels_to_images)
from mmdet.ops import ConvModule, DeformConv
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob
from mmdet.core.bbox import init_pointset_target, refine_pointset_target
from mmdet.ops.minarearect import minaerarect
from mmdet.ops.chamfer_distance import ChamferDistance2D
import math

@HEADS.register_module
class OrientedRepPointsHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 point_feat_channels=256,
                 stacked_convs=3,
                 num_points=9,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_rbox_init=dict(
                     type='IoULoss', loss_weight=0.4),
                 loss_rbox_refine=dict(
                     type='IoULoss', loss_weight=0.75),
                 loss_spatial_init=dict(type='SpatialBorderLoss', loss_weight=0.05),
                 loss_spatial_refine=dict(type='SpatialBorderLoss', loss_weight=0.1),
                 center_init=True,
                 top_ratio=0.4):

        super(OrientedRepPointsHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.point_feat_channels = point_feat_channels
        self.stacked_convs = stacked_convs
        self.num_points = num_points
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss']
        self.loss_cls = build_loss(loss_cls)
        self.loss_rbox_init = build_loss(loss_rbox_init)
        self.loss_rbox_refine = build_loss(loss_rbox_refine)
        self.loss_spatial_init = build_loss(loss_spatial_init)
        self.loss_spatial_refine = build_loss(loss_spatial_refine)
        self.center_init = center_init
        self.top_ratio = top_ratio

        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes
        self.point_generators = [PointGenerator() for _ in self.point_strides]
        # we use deformable conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            
        pts_out_dim = 2 * self.num_points
        self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                             self.point_feat_channels,
                                             self.dcn_kernel, 1, self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_conv = DeformConv(self.feat_channels,
                                                    self.point_feat_channels,
                                                    self.dcn_kernel, 1,
                                                    self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)
        
    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
            
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.reppoints_cls_conv, std=0.01)
        normal_init(self.reppoints_cls_out, std=0.01, bias=bias_cls)
        normal_init(self.reppoints_pts_init_conv, std=0.01)
        normal_init(self.reppoints_pts_init_out, std=0.01)
        normal_init(self.reppoints_pts_refine_conv, std=0.01)
        normal_init(self.reppoints_pts_refine_out, std=0.01)

    def forward_single(self, x):
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        points_init = 0
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))
        pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset)
        cls_out = self.reppoints_cls_out(self.relu(dcn_cls_feat))
        pts_out_refine = self.reppoints_pts_refine_out(self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))

        pts_out_refine = pts_out_refine + pts_out_init.detach()


        return cls_out, pts_out_init, pts_out_refine, x

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_points(self, featmap_sizes, img_metas):
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)
        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(
                featmap_sizes[i], self.point_strides[i])
            multi_level_points.append(points)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]
        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.point_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w = img_meta['pad_shape'][:2]
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
        return points_list, valid_flag_list

    def offset_to_pts(self, center_list, pred_list):
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(-1, 2 * self.num_points)

                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    def neargtcorner(self, pts, gtbboxes):
        gtbboxes = gtbboxes.view(-1, 4, 2)
        pts = pts.view(-1, self.num_points, 2)

        pts_corner_first_ind = ((gtbboxes[:, 0:1, :] - pts) ** 2).sum(dim=2).min(dim=1)[1]
        pts_corner_first_ind = pts_corner_first_ind.reshape(pts_corner_first_ind.shape[0], 1, 1).expand(-1, -1,
                                                                                                        pts.shape[2])
        pts_corner_first = torch.gather(pts, 1, pts_corner_first_ind).squeeze(1)

        pts_corner_sec_ind = ((gtbboxes[:, 1:2, :] - pts) ** 2).sum(dim=2).min(dim=1)[1]
        pts_corner_sec_ind = pts_corner_sec_ind.reshape(pts_corner_sec_ind.shape[0], 1, 1).expand(-1, -1, pts.shape[2])
        pts_corner_sec = torch.gather(pts, 1, pts_corner_sec_ind).squeeze(1)

        pts_corner_third_ind = ((gtbboxes[:, 2:3, :] - pts) ** 2).sum(dim=2).min(dim=1)[1]
        pts_corner_third_ind = pts_corner_third_ind.reshape(pts_corner_third_ind.shape[0], 1, 1).expand(-1, -1,
                                                                                                        pts.shape[2])
        pts_corner_third = torch.gather(pts, 1, pts_corner_third_ind).squeeze(1)

        pts_corner_four_ind = ((gtbboxes[:, 3:4, :] - pts) ** 2).sum(dim=2).min(dim=1)[1]
        pts_corner_four_ind = pts_corner_four_ind.reshape(pts_corner_four_ind.shape[0], 1, 1).expand(-1, -1,
                                                                                                     pts.shape[2])
        pts_corner_four = torch.gather(pts, 1, pts_corner_four_ind).squeeze(1)

        corners = torch.cat([pts_corner_first, pts_corner_sec, pts_corner_third, pts_corner_four], dim=1)
        return corners

    def sampling_points(self, corners, points_num):
        device = corners.device
        corners_xs, corners_ys = corners[:, 0::2], corners[:, 1::2]
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
        all_1_edge_x_points = edge_ratio * first_edge_x_points[:, 1:2] + \
                              (1 - edge_ratio) * first_edge_x_points[:, 0:1]
        all_1_edge_y_points = edge_ratio * first_edge_y_points[:, 1:2] + \
                              (1 - edge_ratio) * first_edge_y_points[:, 0:1]

        all_2_edge_x_points = edge_ratio * sec_edge_x_points[:, 1:2] + \
                              (1 - edge_ratio) * sec_edge_x_points[:, 0:1]
        all_2_edge_y_points = edge_ratio * sec_edge_y_points[:, 1:2] + \
                              (1 - edge_ratio) * sec_edge_y_points[:, 0:1]

        all_3_edge_x_points = edge_ratio * third_edge_x_points[:, 1:2] + \
                              (1 - edge_ratio) * third_edge_x_points[:, 0:1]
        all_3_edge_y_points = edge_ratio * third_edge_y_points[:, 1:2] + \
                              (1 - edge_ratio) * third_edge_y_points[:, 0:1]

        all_4_edge_x_points = edge_ratio * four_edge_x_points_e.unsqueeze(1) + \
                              (1 - edge_ratio) * four_edge_x_points_s.unsqueeze(1)
        all_4_edge_y_points = edge_ratio * four_edge_y_points_e.unsqueeze(1) + \
                              (1 - edge_ratio) * four_edge_y_points_s.unsqueeze(1)

        all_x_points = torch.cat([all_1_edge_x_points, all_2_edge_x_points,
                                  all_3_edge_x_points, all_4_edge_x_points], dim=1).unsqueeze(dim=2)

        all_y_points = torch.cat([all_1_edge_y_points, all_2_edge_y_points,
                                  all_3_edge_y_points, all_4_edge_y_points], dim=1).unsqueeze(dim=2)

        all_points = torch.cat([all_x_points, all_y_points], dim=2)
        return all_points

    def init_loss_single(self,  pts_pred_init, rbox_gt_init, rbox_weights_init, stride):

        normalize_term = self.point_base_scale * stride
        rbox_gt_init = rbox_gt_init.reshape(-1, 8)
        rbox_weights_init = rbox_weights_init.reshape(-1)
        pts_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)
        pos_ind_init = (rbox_weights_init > 0).nonzero().reshape(-1)
        pts_pred_init_norm = pts_pred_init[pos_ind_init]
        rbox_gt_init_norm = rbox_gt_init[pos_ind_init]
        rbox_weights_pos_init = rbox_weights_init[pos_ind_init]
        loss_rbox_init = self.loss_rbox_init(
            pts_pred_init_norm / normalize_term,
            rbox_gt_init_norm / normalize_term,
            rbox_weights_pos_init
        )

        loss_border_init = self.loss_spatial_init(
            pts_pred_init_norm.reshape(-1, 2 * self.num_points) / normalize_term,
            rbox_gt_init_norm / normalize_term,
            rbox_weights_pos_init,
            y_first=False,
            avg_factor=None
        ) if self.loss_spatial_init is not None else loss_rbox_init.new_zeros(1)

        return loss_rbox_init, loss_border_init

    def loss(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             base_features,
             gt_rbboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_rbboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.point_generators)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        # target for initial stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas)
        pts_coordinate_preds_init = self.offset_to_pts(center_list,
                                                       pts_preds_init)

        num_proposals_each_level = [(featmap.size(-1) * featmap.size(-2))
                                    for featmap in cls_scores]

        num_level = len(featmap_sizes)
        assert num_level == len(pts_coordinate_preds_init)
        candidate_list = center_list

        #init_stage assign
        cls_reg_targets_init = init_pointset_target(
            candidate_list,
            valid_flag_list,
            gt_rbboxes,
            img_metas,
            cfg.init,
            gt_rbboxes_ignore_list=gt_rbboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        # get the number of sample of assign)
        (*_, rbbox_gt_list_init, candidate_list_init, rbox_weights_list_init,
         num_total_pos_init, num_total_neg_init, gt_inds_init) = cls_reg_targets_init

        # target for refinement stage
        center_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)
        pts_coordinate_preds_refine = self.offset_to_pts(center_list, pts_preds_refine)

        refine_points_features, = multi_apply(self.get_adaptive_points_feature, base_features, pts_coordinate_preds_refine, self.point_strides)
        features_pts_refine_image = levels_to_images(refine_points_features)
        features_pts_refine_image = [item.reshape(-1, self.num_points, item.shape[-1]) for item in features_pts_refine_image]

        points_list = []
        for i_img, center in enumerate(center_list):
            points = []
            for i_lvl in range(len(pts_preds_refine)):
                points_preds_init_ = pts_preds_init[i_lvl].detach()
                points_preds_init_ = points_preds_init_.view(points_preds_init_.shape[0], -1,
                                                             *points_preds_init_.shape[2:])
                points_shift = points_preds_init_.permute(0, 2, 3, 1) * self.point_strides[i_lvl]
                points_center = center[i_lvl][:, :2].repeat(1, self.num_points)
                points.append(points_center + points_shift[i_img].reshape(-1, 2 * self.num_points))
            points_list.append(points)

        cls_reg_targets_refine = refine_pointset_target(
            points_list,
            valid_flag_list,
            gt_rbboxes,
            img_metas,
            cfg.refine,
            gt_rbboxes_ignore_list=gt_rbboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)

        (labels_list, label_weights_list, rbox_gt_list_refine,
         _, rbox_weights_list_refine, pos_inds_list_refine,
         pos_gt_index_list_refine) = cls_reg_targets_refine

        cls_scores = levels_to_images(cls_scores)
        cls_scores = [
            item.reshape(-1, self.cls_out_channels) for item in cls_scores
        ]

        pts_coordinate_preds_init_image = levels_to_images(
            pts_coordinate_preds_init, flatten=True)
        pts_coordinate_preds_init_image = [
            item.reshape(-1, 2 * self.num_points) for item in pts_coordinate_preds_init_image
        ]

        pts_coordinate_preds_refine_image = levels_to_images(
            pts_coordinate_preds_refine, flatten=True)
        pts_coordinate_preds_refine_image = [
            item.reshape(-1, 2 * self.num_points) for item in pts_coordinate_preds_refine_image
        ]

        with torch.no_grad():

            # refine_stage loc loss
            # quality_assess_list, = multi_apply(self.points_quality_assessment, cls_scores,
            #                                pts_coordinate_preds_refine_image, labels_list,
            #                                rbox_gt_list_refine, label_weights_list,
            #                                rbox_weights_list_refine, pos_inds_list_refine)

            # init stage and refine stage loc loss
            quality_assess_list, = multi_apply(self.points_quality_assessment, features_pts_refine_image, cls_scores,
                                           pts_coordinate_preds_init_image, pts_coordinate_preds_refine_image, labels_list,
                                           rbox_gt_list_refine, label_weights_list,
                                           rbox_weights_list_refine, pos_inds_list_refine)


            labels_list, label_weights_list, rbox_weights_list_refine, num_pos, pos_normalize_term = multi_apply(
                self.point_samples_selection,
                quality_assess_list,
                labels_list,
                label_weights_list,
                rbox_weights_list_refine,
                pos_inds_list_refine,
                pos_gt_index_list_refine,
                num_proposals_each_level=num_proposals_each_level,
                num_level=num_level
            )
            num_pos = sum(num_pos)

        cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
        pts_preds_refine = torch.cat(pts_coordinate_preds_refine_image,
                                     0).view(-1, pts_coordinate_preds_refine_image[0].size(-1))
        labels = torch.cat(labels_list, 0).view(-1)
        labels_weight = torch.cat(label_weights_list, 0).view(-1)
        rbox_gt_refine = torch.cat(rbox_gt_list_refine,
                                    0).view(-1, rbox_gt_list_refine[0].size(-1))
        rbox_weights_refine = torch.cat(rbox_weights_list_refine, 0).view(-1)
        pos_normalize_term = torch.cat(pos_normalize_term, 0).reshape(-1)
        pos_inds_flatten = (labels > 0).nonzero().reshape(-1)
        assert len(pos_normalize_term) == len(pos_inds_flatten)
        if num_pos:
            losses_cls = self.loss_cls(
                cls_scores, labels, labels_weight, avg_factor=num_pos)
            pos_pts_pred_refine = pts_preds_refine[pos_inds_flatten]
            pos_rbox_gt_refine = rbox_gt_refine[pos_inds_flatten]
            pos_rbox_weights_refine = rbox_weights_refine[pos_inds_flatten]
            losses_rbox_refine = self.loss_rbox_refine(
                pos_pts_pred_refine / pos_normalize_term.reshape(-1, 1),
                pos_rbox_gt_refine / pos_normalize_term.reshape(-1, 1),
                pos_rbox_weights_refine
            )

            loss_border_refine = self.loss_spatial_refine(
                pos_pts_pred_refine.reshape(-1, 2 * self.num_points) / pos_normalize_term.reshape(-1, 1),
                pos_rbox_gt_refine / pos_normalize_term.reshape(-1, 1),
                pos_rbox_weights_refine,
                y_first=False,
                avg_factor=None
            ) if self.loss_spatial_refine is not None else losses_rbox_refine.new_zeros(1)

        else:
            losses_cls = cls_scores.sum() * 0
            losses_rbox_refine = pts_preds_refine.sum() * 0
            loss_border_refine = pts_preds_refine.sum() * 0

        losses_rbox_init, loss_border_init = multi_apply(
            self.init_loss_single,
            pts_coordinate_preds_init,
            rbbox_gt_list_init,
            rbox_weights_list_init,
            self.point_strides)

        loss_dict_all = {
            'loss_cls': losses_cls,
            'loss_rbox_init': losses_rbox_init,
            'loss_rbox_refine': losses_rbox_refine,
            'loss_spatial_init': loss_border_init,
            'loss_spatial_refine': loss_border_refine
        }
        return loss_dict_all

    def get_adaptive_points_feature(self, features, locations, stride):
        '''
        features: [b, c, w, h]
        locations: [b, N, 18]
        stride: the stride of the feature map
        '''

        h = features.shape[2] * stride
        w = features.shape[3] * stride

        locations = locations.view(locations.shape[0], locations.shape[1], -1, 2).clone()
        locations[..., 0] = locations[..., 0] / (w / 2.) - 1
        locations[..., 1] = locations[..., 1] / (h / 2.) - 1

        batch_size = features.size(0)
        sampled_features = torch.zeros([locations.shape[0],
                                        features.size(1),
                                        locations.size(1),
                                        locations.size(2)
                                        ]).to(locations.device)

        for i in range(batch_size):
            feature = nn.functional.grid_sample(features[i:i + 1], locations[i:i + 1])[0]
            sampled_features[i] = feature

        return sampled_features,

    def points_quality_assessment(self, points_features, cls_score, pts_pred_init, pts_pred_refine, label, rbbox_gt, label_weight, rbox_weight, pos_inds):

        pos_scores = cls_score[pos_inds]
        pos_pts_pred_init = pts_pred_init[pos_inds]
        pos_pts_pred_refine = pts_pred_refine[pos_inds]
        pos_pts_refine_features = points_features[pos_inds]

        pos_rbbox_gt = rbbox_gt[pos_inds]
        pos_label = label[pos_inds]
        pos_label_weight = label_weight[pos_inds]
        pos_rbox_weight = rbox_weight[pos_inds]

        pts_feats_dissimilarity = self.feature_cosine_similarity(pos_pts_refine_features)

        qua_cls = self.loss_cls(
            pos_scores,
            pos_label,
            pos_label_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        corners_pred_init = minaerarect(pos_pts_pred_init)
        corners_pred_refine = minaerarect(pos_pts_pred_refine)
        # corners_pred = self.neargtcorner(pos_pts_pred, pos_rbbox_gt)

        sampling_pts_pred_init = self.sampling_points(corners_pred_init, 10)
        sampling_pts_pred_refine = self.sampling_points(corners_pred_refine, 10)
        corners_pts_gt = self.sampling_points(pos_rbbox_gt, 10)

        qua_ori_init = ChamferDistance2D(corners_pts_gt, sampling_pts_pred_init)
        qua_ori_refine = ChamferDistance2D(corners_pts_gt, sampling_pts_pred_refine)

        qua_loc_init = self.loss_rbox_refine(
            pos_pts_pred_init,
            pos_rbbox_gt,
            pos_rbox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        qua_loc_refine = self.loss_rbox_refine(
            pos_pts_pred_refine,
            pos_rbbox_gt,
            pos_rbox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        qua_cls = qua_cls.sum(-1)
        # weight inti-stage and refine-stage
        qua = qua_cls + 0.2*(qua_loc_init + 0.3 * qua_ori_init) + 0.8 * (
                    qua_loc_refine + 0.3 * qua_ori_refine) + 0.1*pts_feats_dissimilarity

        return qua,


    def feature_cosine_similarity(self, points_features):
        '''
        points_features: [N_pos, 9, 256]
        '''
        # print('points_features', points_features.shape)
        mean_points_feats = torch.mean(points_features, dim=1, keepdim=True)
        # print('mean_points_feats', mean_points_feats.shape)

        norm_pts_feats = torch.norm(points_features, p=2, dim=2).unsqueeze(dim=2).clamp(min=1e-2)
        # print('norm_pts_feats', norm_pts_feats)
        norm_mean_pts_feats = torch.norm(mean_points_feats, p=2, dim=2).unsqueeze(dim=2).clamp(min=1e-2)
        # print('norm_mean_pts_feats', norm_mean_pts_feats)

        unity_points_features = points_features / norm_pts_feats


        unity_mean_points_feats = mean_points_feats / norm_mean_pts_feats

        cos_similarity = nn.CosineSimilarity(dim=2, eps=1e-6)
        feats_similarity = 1.0 - cos_similarity(unity_points_features, unity_mean_points_feats)
        # print('feats_similarity', feats_similarity)

        max_correlation, _ = torch.max(feats_similarity, dim=1)
        # print('max_correlation', max_correlation.shape, max_correlation)
        return max_correlation

    def point_samples_selection(self, quality_assess, label, label_weight, rbox_weight,
                     pos_inds, pos_gt_inds, num_proposals_each_level=None, num_level=None):
        '''
              The selection of point set samples based on the quality assessment values.
        '''

        if len(pos_inds) == 0:
            return label, label_weight, rbox_weight, 0, torch.tensor([]).type_as(rbox_weight)

        num_gt = pos_gt_inds.max()
        num_proposals_each_level_ = num_proposals_each_level.copy()
        num_proposals_each_level_.insert(0, 0)
        inds_level_interval = np.cumsum(num_proposals_each_level_)
        pos_level_mask = []
        for i in range(num_level):
            mask = (pos_inds >= inds_level_interval[i]) & (
                    pos_inds < inds_level_interval[i + 1])
            pos_level_mask.append(mask)
        pos_inds_after_select = []
        ignore_inds_after_select = []

        for gt_ind in range(num_gt):
            pos_inds_select = []
            pos_loss_select = []
            gt_mask = pos_gt_inds == (gt_ind + 1)
            for level in range(num_level):
                level_mask = pos_level_mask[level]
                level_gt_mask = level_mask & gt_mask
                value, topk_inds = quality_assess[level_gt_mask].topk(
                    min(level_gt_mask.sum(), 6), largest=False)
                pos_inds_select.append(pos_inds[level_gt_mask][topk_inds])
                pos_loss_select.append(value)
            pos_inds_select = torch.cat(pos_inds_select)
            pos_loss_select = torch.cat(pos_loss_select)

            if len(pos_inds_select) < 2:
                pos_inds_after_select.append(pos_inds_select)
                ignore_inds_after_select.append(pos_inds_select.new_tensor([]))
            else:
                pos_loss_select, sort_inds = pos_loss_select.sort() # small to large
                pos_inds_select = pos_inds_select[sort_inds]
                topk = math.ceil(pos_loss_select.shape[0] * self.top_ratio)
                pos_inds_select_topk = pos_inds_select[:topk]
                pos_inds_after_select.append(pos_inds_select_topk)
                ignore_inds_after_select.append(pos_inds_select_topk.new_tensor([]))

        pos_inds_after_select = torch.cat(pos_inds_after_select)
        ignore_inds_after_select = torch.cat(ignore_inds_after_select)

        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_select).all(1)
        reassign_ids = pos_inds[reassign_mask]
        label[reassign_ids] = 0
        # label[reassign_ids] = self.num_classes
        label_weight[ignore_inds_after_select] = 0
        rbox_weight[reassign_ids] = 0
        num_pos = len(pos_inds_after_select)

        pos_level_mask_after_select = []
        for i in range(num_level):
            mask = (pos_inds_after_select >= inds_level_interval[i]) & (
                    pos_inds_after_select < inds_level_interval[i + 1])
            pos_level_mask_after_select.append(mask)
        pos_level_mask_after_select = torch.stack(pos_level_mask_after_select, 0).type_as(label)
        pos_normalize_term = pos_level_mask_after_select * (
                self.point_base_scale *
                torch.as_tensor(self.point_strides).type_as(label)).reshape(-1, 1)
        pos_normalize_term = pos_normalize_term[pos_normalize_term > 0].type_as(rbox_weight)
        assert len(pos_normalize_term) == len(pos_inds_after_select)

        return label, label_weight, rbox_weight, num_pos, pos_normalize_term

    def get_bboxes(self,
                   cls_scores,
                   pts_preds_init,
                   pts_preds_refine,
                   base_feats,
                   img_metas,
                   cfg,
                   rescale=False,
                   nms=True):
        assert len(cls_scores) == len(pts_preds_refine)
        
        num_levels = len(cls_scores)
        mlvl_points = [
            self.point_generators[i].grid_points(cls_scores[i].size()[-2:],
                                                 self.point_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            points_pred_list = [
                pts_preds_refine[i][img_id].detach()
                for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, points_pred_list,
                                               mlvl_points, img_shape,
                                               scale_factor, cfg, rescale, nms)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          points_preds,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False,
                          nms=True):
        assert len(cls_scores) == len(points_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_reppoints = []
        
        for i_lvl, (cls_score, points_pred, points) in enumerate(
                zip(cls_scores, points_preds, mlvl_points)):
            assert cls_score.size()[-2:] == points_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            points_pred = points_pred.permute(1, 2, 0).reshape(-1, 2 * self.num_points)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                points_pred = points_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            pts_pred = points_pred.reshape(-1, self.num_points, 2)
            pts_pred_offsety = pts_pred[:, :, 0::2]
            pts_pred_offsetx = pts_pred[:, :, 1::2]
            pts_pred = torch.cat([pts_pred_offsetx, pts_pred_offsety], dim=2).reshape(-1, 2 * self.num_points)
            bbox_pred = minaerarect(pts_pred)

            bbox_pos_center = points[:, :2].repeat(1, 4)
            bboxes = bbox_pred * self.point_strides[i_lvl] + bbox_pos_center
            
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

            points_pred = points_pred.reshape(-1, self.num_points, 2)
            points_pred_dy = points_pred[:, :, 0::2]
            points_pred_dx = points_pred[:, :, 1::2]
            pts = torch.cat([points_pred_dx, points_pred_dy], dim=2).reshape(-1, 2 * self.num_points)

            pts_pos_center = points[:, :2].repeat(1, self.num_points)
            pts = pts * self.point_strides[i_lvl] + pts_pos_center

            mlvl_reppoints.append(pts)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_reppoints = torch.cat(mlvl_reppoints)

        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_reppoints /= mlvl_reppoints.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        if nms:
            det_bboxes, det_labels = multiclass_rnms(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img, multi_reppoints=mlvl_reppoints)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores


