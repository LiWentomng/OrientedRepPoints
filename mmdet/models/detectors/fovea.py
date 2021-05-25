from ..registry import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module
class FOVEA(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FOVEA, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained)
