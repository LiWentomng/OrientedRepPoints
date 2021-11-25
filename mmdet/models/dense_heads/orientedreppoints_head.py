from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.cnn import constant_init
from mmdet.core import (PointGenerator, multi_apply, multiclass_obbnms,
                       levels_to_images)
from mmdet.ops import ConvModule, DeformConv
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob
from mmdet.core.bbox import init_ori_pointset_target, refine_ori_pointset_target
from mmdet.ops.minarearect import minaerarect
from mmdet.ops.chamfer_distance import ChamferDistance2D
import math

@HEADS.register_module
class OrientedRepPointsHead(nn.Module):
    """
    Oriented RepPoints head.

    Args:
        num_classes (int): Number of categories of aerial dataset.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        point_feat_channels (int): Number of channels of points features.
        stacked_convs (int): How many conv layers are used.
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Iterable): points strides.
        point_base_scale (int): bbox scale for assigning labels.

        loss_cls (dict): Config of classification loss.
        loss_obox_init (dict): Config of oriented reppoints loss in initialization stage.
        loss_obox_refine (dict): Config of oriented reppoints loss in refinement stage.
        loss_spatial_init (dict): Config of spatial constraint loss in initialization stage.
        loss_spatial_refine (dict): Config of spatial constraint loss in refinement stage.
        center_init (bool): Whether to use center point assignment.
        top_ratio: top sampling ratio in dynamic top k reassign of APAA.
        qua_init_w: intial weight of Q in APAA.
        qua_refine_w: refinement weight of Q in APAA.
    """

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
                 loss_obox_init=dict(
                     type='IoULoss', loss_weight=0.4),
                 loss_obox_refine=dict(
                     type='IoULoss', loss_weight=0.75),
                 loss_spatial_init=dict(type='SpatialBorderLoss', loss_weight=0.05),
                 loss_spatial_refine=dict(type='SpatialBorderLoss', loss_weight=0.1),
                 center_init=True,
                 top_ratio=0.4,
                 qua_init_w=0.2,
                 qua_refine_w=0.8):

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
        self.loss_obox_init = build_loss(loss_obox_init)
        self.loss_obox_refine = build_loss(loss_obox_refine)
        self.loss_spatial_init = build_loss(loss_spatial_init)
        self.loss_spatial_refine = build_loss(loss_spatial_refine)
        self.center_init = center_init
        self.top_ratio = top_ratio
        self.init_qua_w = qua_init_w
        self.refine_qua_w = qua_refine_w

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
        self.oreppoints_cls_conv = DeformConv(self.feat_channels,
                                             self.point_feat_channels,
                                             self.dcn_kernel, 1, self.dcn_pad)
        self.oreppoints_cls_out = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)
        self.oreppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.oreppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)
        self.oreppoints_pts_refine_conv = DeformConv(self.feat_channels,
                                                    self.point_feat_channels,
                                                    self.dcn_kernel, 1,
                                                    self.dcn_pad)
        self.oreppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)
        
    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
            
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.oreppoints_cls_conv, std=0.01)
        normal_init(self.oreppoints_cls_out, std=0.01, bias=bias_cls)
        normal_init(self.oreppoints_pts_init_conv, std=0.01)
        normal_init(self.oreppoints_pts_init_out, std=0.01)
        normal_init(self.oreppoints_pts_refine_conv, std=0.01)
        normal_init(self.oreppoints_pts_refine_out, std=0.01)

    def forward_single(self, x):
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        points_init = 0
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        # initialize oriented reppoints
        pts_out_init = self.oreppoints_pts_init_out(
            self.relu(self.oreppoints_pts_init_conv(pts_feat)))
        pts_out_init = pts_out_init + points_init
        # refine and classify oriented reppoints
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        dcn_cls_feat = self.oreppoints_cls_conv(cls_feat, dcn_offset)
        cls_out = self.oreppoints_cls_out(self.relu(dcn_cls_feat))
        pts_out_refine = self.oreppoints_pts_refine_out(self.relu(self.oreppoints_pts_refine_conv(pts_feat, dcn_offset)))

        pts_out_refine = pts_out_refine + pts_out_init.detach()
        return cls_out, pts_out_init, pts_out_refine, x

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_points(self, featmap_sizes, img_metas):
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)
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
        """
        Convert from point offset to point coordinate.
        """
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

    def nearestgtcorner(self, pts, gt_rboxes):
        '''
         oriented conversion func.: nearestgtcorner.
        :param pts: adaptive points  shape:[n, pt_num*2].
        :param gtrboxes: the assigned ground truth box shape:[n, 4*2].
        :return: the converted oriented bounding boxes.
        '''

        gt_rboxes = gt_rboxes.view(-1, 4, 2)
        pts = pts.view(-1, self.num_points, 2)

        obb_pt1_ind = ((gt_rboxes[:, 0:1, :] - pts) ** 2).sum(dim=2).min(dim=1)[1]
        obb_pt1_ind = obb_pt1_ind.reshape(obb_pt1_ind.shape[0], 1, 1).expand(-1, -1, pts.shape[2])
        obb_pt1 = torch.gather(pts, 1, obb_pt1_ind).squeeze(1)

        obb_pt2_ind = ((gt_rboxes[:, 1:2, :] - pts) ** 2).sum(dim=2).min(dim=1)[1]
        obb_pt2_ind = obb_pt2_ind.reshape(obb_pt2_ind.shape[0], 1, 1).expand(-1, -1, pts.shape[2])
        obb_pt2 = torch.gather(pts, 1, obb_pt2_ind).squeeze(1)

        obb_pt3_ind = ((gt_rboxes[:, 2:3, :] - pts) ** 2).sum(dim=2).min(dim=1)[1]
        obb_pt3_ind = obb_pt3_ind.reshape(obb_pt3_ind.shape[0], 1, 1).expand(-1, -1, pts.shape[2])
        obb_pt3 = torch.gather(pts, 1, obb_pt3_ind).squeeze(1)

        obb_pt4_ind = ((gt_rboxes[:, 3:4, :] - pts) ** 2).sum(dim=2).min(dim=1)[1]
        obb_pt4_ind = obb_pt4_ind.reshape(obb_pt4_ind.shape[0], 1, 1).expand(-1, -1, pts.shape[2])
        obb_pt4 = torch.gather(pts, 1, obb_pt4_ind).squeeze(1)

        obbexes = torch.cat([obb_pt1, obb_pt2, obb_pt3, obb_pt4], dim=1)

        return obbexes


    def obb_sampling_points(self, obboxes, points_num):
        '''
        sample the edge points for the orientation quality assessment.
        param obboxes: the corner points of the oriented bounding box.
        param points_num: the number of samping points.
        return: the sampling points of the whole oriented bounding box.
        '''

        device = obboxes.device
        obboxes_xs, obboxes_ys = obboxes[:, 0::2], obboxes[:, 1::2]

        edge1_pts_x, edge1_pts_y = obboxes_xs[:, 0:2], obboxes_ys[:, 0:2]
        edge2_pts_x, edge2_pts_y = obboxes_xs[:, 1:3], obboxes_ys[:, 1:3]
        edge3_pts_x, edge3_pts_y = obboxes_xs[:, 2:4], obboxes_ys[:, 2:4]
        edge4_pts_s_x, edge4_pts_s_y = obboxes_xs[:, 3], obboxes_ys[:, 3]
        edge4_pts_e_x, edge4_pts_e_y = obboxes_xs[:, 0], obboxes_ys[:, 0]

        #sampling ratio
        ratio = torch.linspace(0, 1, points_num).to(device).repeat(obboxes.shape[0], 1)

        pts_edge1_x = ratio * edge1_pts_x[:, 1:2] + (1 - ratio) * edge1_pts_x[:, 0:1]
        pts_edge1_y = ratio * edge1_pts_y[:, 1:2] + (1 - ratio) * edge1_pts_y[:, 0:1]
        pts_edge2_x = ratio * edge2_pts_x[:, 1:2] + (1 - ratio) * edge2_pts_x[:, 0:1]
        pts_edge2_y = ratio * edge2_pts_y[:, 1:2] + (1 - ratio) * edge2_pts_y[:, 0:1]
        pts_edge3_x = ratio * edge3_pts_x[:, 1:2] + (1 - ratio) * edge3_pts_x[:, 0:1]
        pts_edge3_y = ratio * edge3_pts_y[:, 1:2] + (1 - ratio) * edge3_pts_y[:, 0:1]
        pts_edge4_x = ratio * edge4_pts_e_x.unsqueeze(1) + (1 - ratio) * edge4_pts_s_x.unsqueeze(1)
        pts_edge4_y = ratio * edge4_pts_e_y.unsqueeze(1) + (1 - ratio) * edge4_pts_s_y.unsqueeze(1)

        pts_x = torch.cat([pts_edge1_x, pts_edge2_x, pts_edge3_x, pts_edge4_x], dim=1).unsqueeze(dim=2)
        pts_y = torch.cat([pts_edge1_y, pts_edge2_y, pts_edge3_y, pts_edge4_y], dim=1).unsqueeze(dim=2)

        sampling_poly_points = torch.cat([pts_x, pts_y], dim=2)

        return sampling_poly_points


    def get_adaptive_points_feature(self, features, pts_locations, stride):
        '''
        interpret the features of learning adaptive points
        features: [b, c, w, h]
        pts_locations: [b, N, pt_num*2]
        stride: the stride of the feture map
        return: point-wise feature vetors (b, 256, n, pt_num)
        '''

        h = features.shape[2] * stride
        w = features.shape[3] * stride

        pts_locations = pts_locations.view(pts_locations.shape[0], pts_locations.shape[1], -1, 2).clone()
        pts_locations[..., 0] = pts_locations[..., 0] / (w / 2.) - 1
        pts_locations[..., 1] = pts_locations[..., 1] / (h / 2.) - 1

        batch_size = features.size(0)
        sampled_features = torch.zeros([pts_locations.shape[0],  # b
                                        features.size(1),  # c
                                        pts_locations.size(1),  # N
                                        pts_locations.size(2)]).to(pts_locations.device)  # pts_num

        for i in range(batch_size):
            feature = nn.functional.grid_sample(features[i:i + 1], pts_locations[i: i + 1])[0]
            sampled_features[i] = feature

        return sampled_features,

    def feature_cosine_similarity(self, points_features):
        '''
        the point-wise correlation of learning adaptive points
        '''

        mean_points_feats = torch.mean(points_features, dim=1, keepdim=True)
        norm_pts_feats = torch.norm(points_features, p=2, dim=2).unsqueeze(dim=2).clamp(min=1e-2)
        norm_mean_feats = torch.norm(mean_points_feats, p=2, dim=2).unsqueeze(dim=2).clamp(min=1e-2)

        unity_points_features = points_features / norm_pts_feats
        unity_mean_points_feats = mean_points_feats / norm_mean_feats

        # features similarity
        cos_similarity = nn.CosineSimilarity(dim=2, eps=1e-6)
        mean_cos_similarity = torch.mean(cos_similarity(unity_points_features, unity_mean_points_feats), dim=1)
        feats_dissimilarity = 1.0 - mean_cos_similarity

        return feats_dissimilarity

    def points_quality_assessment(self, pts_feats, cls_score, pts_pred_init, pts_pred_refine, label, obbox_gt, label_weight, obox_weight, pos_inds):
        '''
        quality assessment of learning adaptive points in APAA.
        '''
        pos_scores = cls_score[pos_inds]
        pos_pts_pred_init = pts_pred_init[pos_inds]
        pos_pts_pred_refine = pts_pred_refine[pos_inds]
        pos_pts_refine_feats = pts_feats[pos_inds]

        pos_obbox_gt = obbox_gt[pos_inds]
        pos_label = label[pos_inds]
        pos_label_weight = label_weight[pos_inds]
        pos_obox_weight = obox_weight[pos_inds]

        qua_cls = self.loss_cls(
            pos_scores,
            pos_label,
            pos_label_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')
        # classification quality
        qua_cls = qua_cls.sum(-1)

        obbox_pred_init = minaerarect(pos_pts_pred_init)
        obbox_pred_refine = minaerarect(pos_pts_pred_refine)
        sampling_pts_pred_init = self.obb_sampling_points(obbox_pred_init, 10)
        sampling_pts_pred_refine = self.obb_sampling_points(obbox_pred_refine, 10)
        sampling_pts_gt = self.obb_sampling_points(pos_obbox_gt, 10)

        # orientation quality
        qua_ori_init = ChamferDistance2D(sampling_pts_gt, sampling_pts_pred_init)
        qua_ori_refine = ChamferDistance2D(sampling_pts_gt, sampling_pts_pred_refine)

        #localizaiton quality
        qua_loc_init = self.loss_obox_refine(
            pos_pts_pred_init,
            pos_obbox_gt,
            pos_obox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')
        qua_loc_refine = self.loss_obox_refine(
            pos_pts_pred_refine,
            pos_obbox_gt,
            pos_obox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        # point-wise correlation quality
        pts_feats_dissimilarity = self.feature_cosine_similarity(pos_pts_refine_feats)

        # weighted inti-stage and refine-stage
        qua_assess = qua_cls + self.init_qua_w *(qua_loc_init + qua_ori_init) \
              + self.refine_qua_w * (qua_loc_refine + qua_ori_refine) + pts_feats_dissimilarity

        return qua_assess,

    def dynamic_topk_reassign(self, quality_assess, label, label_weight, obox_weight,
                     pos_inds, pos_gt_inds, num_proposals_each_level=None, num_level=None):
        '''
        dynamic top k selection and reassign of points samples
        based on the quality assessment in APAA.
        '''

        if len(pos_inds) == 0:
            return label, label_weight, obox_weight, 0, torch.tensor([]).type_as(obox_weight)

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

                #dynamic topk selection
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
        obox_weight[reassign_ids] = 0
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
        pos_normalize_term = pos_normalize_term[pos_normalize_term > 0].type_as(obox_weight)
        assert len(pos_normalize_term) == len(pos_inds_after_select)

        return label, label_weight, obox_weight, num_pos, pos_normalize_term


    def init_loss_single(self,  pts_pred_init, obox_gt_init, obox_weights_init, stride):
        '''
        localization loss of initialization stage
        '''

        normalize_term = self.point_base_scale * stride
        obox_gt_init = obox_gt_init.reshape(-1, 8)
        obox_weights_init = obox_weights_init.reshape(-1)
        pts_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)
        pos_ind_init = (obox_weights_init > 0).nonzero().reshape(-1)
        pts_pred_init_norm = pts_pred_init[pos_ind_init]
        obox_gt_init_norm = obox_gt_init[pos_ind_init]
        obox_weights_pos_init = obox_weights_init[pos_ind_init]

        # initial localization loss
        loss_obox_init = self.loss_obox_init(
            pts_pred_init_norm / normalize_term,
            obox_gt_init_norm / normalize_term,
            obox_weights_pos_init
        )

        # initial spatial constraint loss
        loss_spatial_init = self.loss_spatial_init(
            pts_pred_init_norm.reshape(-1, 2 * self.num_points) / normalize_term,
            obox_gt_init_norm / normalize_term,
            obox_weights_pos_init,
            y_first=False,
            avg_factor=None
        ) if self.loss_spatial_init is not None else loss_obox_init.new_zeros(1)

        return loss_obox_init, loss_spatial_init

    def loss(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             base_features,
             gt_obboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_oboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.point_generators)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        # targets for initial stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,img_metas)
        pts_coordinate_preds_init = self.offset_to_pts(center_list, pts_preds_init)

        num_proposals_each_level = [(featmap.size(-1) * featmap.size(-2))
                                    for featmap in cls_scores]

        num_level = len(featmap_sizes)
        assert num_level == len(pts_coordinate_preds_init)
        candidate_list = center_list

        #init_stage assign
        cls_reg_targets_init = init_ori_pointset_target(
            candidate_list,
            valid_flag_list,
            gt_obboxes,
            img_metas,
            cfg.init,
            gt_obboxes_ignore_list=gt_oboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)

        # get the number of sample of assign)
        (*_, obbox_gt_list_init, candidate_list_init, obbox_weights_list_init,
         num_total_pos_init, num_total_neg_init, gt_inds_init) = cls_reg_targets_init

        # target for refinement stage
        center_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)
        pts_coordinate_preds_refine = self.offset_to_pts(center_list, pts_preds_refine)

        #adaptive points features
        refine_points_features, = multi_apply(self.get_adaptive_points_feature, base_features,
                                              pts_coordinate_preds_refine, self.point_strides)
        features_pts_refine_image = levels_to_images(refine_points_features)
        features_pts_refine_image = [item.reshape(-1, self.num_points, item.shape[-1]) for item in
                                     features_pts_refine_image]

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

        cls_reg_targets_refine = refine_ori_pointset_target(
            points_list,
            valid_flag_list,
            gt_obboxes,
            img_metas,
            cfg.refine,
            gt_obboxes_ignore_list=gt_oboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)

        (labels_list, label_weights_list, obox_gt_list_refine,
         _, obox_weights_list_refine, pos_inds_list_refine,
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

            # quality assessment of learning adaptive points
            quality_assess_list, = multi_apply(self.points_quality_assessment, features_pts_refine_image, cls_scores,
                                           pts_coordinate_preds_init_image, pts_coordinate_preds_refine_image, labels_list,
                                           obox_gt_list_refine, label_weights_list,
                                           obox_weights_list_refine, pos_inds_list_refine)


            labels_list, label_weights_list, obox_weights_list_refine, num_pos, pos_normalize_term = multi_apply(
                self.dynamic_topk_reassign,
                quality_assess_list,
                labels_list,
                label_weights_list,
                obox_weights_list_refine,
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
        obox_gt_refine = torch.cat(obox_gt_list_refine,
                                    0).view(-1, obox_gt_list_refine[0].size(-1))
        obox_weights_refine = torch.cat(obox_weights_list_refine, 0).view(-1)
        pos_normalize_term = torch.cat(pos_normalize_term, 0).reshape(-1)
        pos_inds_flatten = (labels > 0).nonzero().reshape(-1)
        assert len(pos_normalize_term) == len(pos_inds_flatten)

        if num_pos:
            # classification loss
            losses_cls = self.loss_cls(
                cls_scores, labels, labels_weight, avg_factor=num_pos)
            pos_pts_pred_refine = pts_preds_refine[pos_inds_flatten]
            pos_obox_gt_refine = obox_gt_refine[pos_inds_flatten]
            pos_obox_weights_refine = obox_weights_refine[pos_inds_flatten]

            # localization loss in refine stage
            losses_obox_refine = self.loss_obox_refine(
                pos_pts_pred_refine / pos_normalize_term.reshape(-1, 1),
                pos_obox_gt_refine / pos_normalize_term.reshape(-1, 1),
                pos_obox_weights_refine
            )

            # spatial constraint loss in refine stage
            loss_spatial_refine = self.loss_spatial_refine(
                pos_pts_pred_refine.reshape(-1, 2 * self.num_points) / pos_normalize_term.reshape(-1, 1),
                pos_obox_gt_refine / pos_normalize_term.reshape(-1, 1),
                pos_obox_weights_refine,
                y_first=False,
                avg_factor=None
            ) if self.loss_spatial_refine is not None else losses_obox_refine.new_zeros(1)

        else:
            losses_cls = cls_scores.sum() * 0
            losses_obox_refine = pts_preds_refine.sum() * 0
            loss_spatial_refine = pts_preds_refine.sum() * 0

        losses_obox_init, loss_spatial_init = multi_apply(
            self.init_loss_single,
            pts_coordinate_preds_init,
            obbox_gt_list_init,
            obbox_weights_list_init,
            self.point_strides)

        loss_dict_all = {
            'loss_cls': losses_cls,
            'loss_obox_init': losses_obox_init,
            'loss_obox_refine': losses_obox_refine,
            'loss_spatial_init': loss_spatial_init,
            'loss_spatial_refine': loss_spatial_refine
        }
        return loss_dict_all

    def get_obboxes(self,
                   cls_scores,
                   pts_preds_init,
                   pts_preds_refine,
                   base_feats,
                   img_metas,
                   cfg,
                   rescale=False,
                   nms=True):
        '''
        get the oriented bounding boxes based on the predicted oriented reppoints
        '''

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
            ori_proposals = self.get_oboxes_single(cls_score_list, points_pred_list,
                                               mlvl_points, img_shape,
                                               scale_factor, cfg, rescale, nms)
            result_list.append(ori_proposals)
        return result_list


    def get_oboxes_single(self,
                          cls_scores,
                          points_preds,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False,
                          nms=True):

        assert len(cls_scores) == len(points_preds) == len(mlvl_points)
        mlvl_oboxes = []
        mlvl_scores = []
        mlvl_oreppoints = []

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
            obox_pred = minaerarect(pts_pred)

            bbox_pos_center = points[:, :2].repeat(1, 4)
            oboxes = obox_pred * self.point_strides[i_lvl] + bbox_pos_center

            mlvl_oboxes.append(oboxes)
            mlvl_scores.append(scores)

            points_pred = points_pred.reshape(-1, self.num_points, 2)
            points_pred_dy = points_pred[:, :, 0::2]
            points_pred_dx = points_pred[:, :, 1::2]
            pts = torch.cat([points_pred_dx, points_pred_dy], dim=2).reshape(-1, 2 * self.num_points)

            pts_pos_center = points[:, :2].repeat(1, self.num_points)
            pts = pts * self.point_strides[i_lvl] + pts_pos_center

            mlvl_oreppoints.append(pts)
        mlvl_oboxes = torch.cat(mlvl_oboxes)
        mlvl_oreppoints = torch.cat(mlvl_oreppoints)

        if rescale:
            mlvl_oboxes /= mlvl_oboxes.new_tensor(scale_factor)
            mlvl_oreppoints /= mlvl_oreppoints.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        if nms:
            det_oboxes, det_labels = multiclass_obbnms(mlvl_oboxes, mlvl_scores,
                                                     cfg.score_thr, cfg.nms,
                                                     cfg.max_per_img, multi_oreppoints=mlvl_oreppoints)
            return det_oboxes, det_labels
        else:
            return mlvl_oboxes, mlvl_scores

