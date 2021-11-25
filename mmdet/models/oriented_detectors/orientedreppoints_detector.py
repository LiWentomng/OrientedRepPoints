from ..registry import DETECTORS
from .single_stage import SingleStageOBBDetector
from mmdet.core import obbox2result, merge_aug_poly_results
from mmdet.core import multiclass_obbnms

@DETECTORS.register_module
class OrientedRepPointsDetector(SingleStageOBBDetector):

    """
    Oriented RepPoints for Aerial Object Detection.
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
                      gt_bboxes_ignore=None):
        """
        forward train in orientedreppoints_head.py
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_oboxes_ignore=gt_bboxes_ignore)
        return losses


    def simple_test(self, img, img_metas, rescale=False):
        """
        simple test to get predicted oriented reppoints and oriented bounding boxes
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        obox_inputs = outs + (img_metas, self.test_cfg, rescale)
        obox_list = self.bbox_head.get_obboxes(*obox_inputs)
        obox_results = [
            obbox2result(det_oboxes, det_labels, self.bbox_head.num_classes)
            for det_oboxes, det_labels in obox_list
        ]
        return obox_results[0]


    def aug_test(self, imgs, img_metas, rescale=False):
        """
        augmented detection obboxes and scores
        """
        feats = self.extract_feats(imgs)
        aug_obboxes = []
        aug_scores = []

        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.bbox_head(x)
            obbox_inputs = outs + (img_meta, self.test_cfg, False, False)
            det_obboxes, det_scores = self.bbox_head.get_obboxes(*obbox_inputs)[0]
            aug_obboxes.append(det_obboxes)
            aug_scores.append(det_scores)

        # after merging, obboxes will be rescaled to the original image size
        merged_obboxes, merged_scores = merge_aug_poly_results(
            aug_obboxes, aug_scores, img_metas)

        det_obboxes, det_labels = multiclass_obbnms(merged_obboxes, merged_scores,
                                                 self.test_cfg.score_thr, self.test_cfg.nms,
                                                 self.test_cfg.max_per_img)

        if rescale:
            _det_obboxes = det_obboxes
        else:
            _det_obboxes = det_obboxes.clone()
            _det_obboxes[:, :8] *= img_metas[0][0]['scale_factor']
        obbox_results = obbox2result(_det_obboxes, det_labels,
                                   self.bbox_head.num_classes)
        return obbox_results

