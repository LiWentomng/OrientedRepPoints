from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from mmdet.core import rbbox2result

import torch
from mmdet.core import multiclass_rnms

@DETECTORS.register_module
class OrientedRepPointsDetector(SingleStageDetector):
    """ Oriented RepPoints: Point Set Representation for Aerial Object Detection.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(OrientedRepPointsDetector,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_rbboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_rbboxes_ignore=gt_rbboxes_ignore)
        return losses
    
    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_metas, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]


    def rbbox_flip(self, rbboxes, img_shape, direction='horizontal'):
        """Flip bboxes horizontally.
        Args:
            rbboxes (list[Tensor]): shape (..., 8*k)
            img_shape(tuple): (height, width)
        """
        assert rbboxes.shape[-1] % 8 == 0
        flipped = rbboxes.clone()
        # print('img_shape', img_shape)
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::8] = w - rbboxes[..., 0::8] - 1
            flipped[..., 2::8] = w - rbboxes[..., 2::8] - 1
            flipped[..., 4::8] = w - rbboxes[..., 4::8] - 1
            flipped[..., 6::8] = w - rbboxes[..., 6::8] - 1
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::8] = h - rbboxes[..., 1::8] - 1
            flipped[..., 3::8] = h - rbboxes[..., 3::8] - 1
            flipped[..., 5::8] = h - rbboxes[..., 5::8] - 1
            flipped[..., 7::8] = h - rbboxes[..., 7::8] - 1
        else:
            raise ValueError(
                'Invalid flipping direction "{}"'.format(direction))
        return flipped

    def rbox_mapping_back(self, rboxes, img_shape, scale_factor, flip):
        fliped_bboxes = self.rbbox_flip(rboxes, img_shape) if flip else rboxes
        fliped_bboxes = fliped_bboxes / scale_factor
        return fliped_bboxes


    def merge_aug_results(self, aug_bboxes, aug_scores, img_metas):
        """
        Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]) : shape (n, 8*#class)
            aug_scores (list[Tensor]): shape (n, #class)

        Returns:
            tuple: (bboxes, scores)
        """

        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            # print('bboxes', bboxes.shape)
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            # flip_direction = img_info[0]['flip_direction']
            # rotate_angle = img_info[0]['rotate_angle']
            bboxes = self.rbox_mapping_back(bboxes, img_shape, scale_factor, flip)
            recovered_bboxes.append(bboxes)

        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores

    def aug_test(self, imgs, img_metas, rescale=False):
        # print('aug')
        feats = self.extract_feats(imgs)

        aug_bboxes = []
        aug_scores = []

        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.bbox_head(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, False, False)
            # print('bbox_inputs', len(bbox_inputs))
            det_bboxes, det_scores = self.bbox_head.get_bboxes(*bbox_inputs)[0]
            # print('det_bboxes', det_bboxes.shape) # [N,8]
            # print('det_scores', det_scores.shape) # [N, 16]
            aug_bboxes.append(det_bboxes)
            aug_scores.append(det_scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = self.merge_aug_results(
            aug_bboxes, aug_scores, img_metas)

        det_bboxes, det_labels = multiclass_rnms(merged_bboxes, merged_scores,
                                                 self.test_cfg.score_thr, self.test_cfg.nms,
                                                 self.test_cfg.max_per_img)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :8] *= img_metas[0][0]['scale_factor']
        bbox_results = rbbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        return bbox_results
        
