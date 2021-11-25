from .coco import CocoDataset
from .registry import DATASETS
import numpy as np

@DATASETS.register_module
class DiorDataset(CocoDataset):

    CLASSES = ("airplane", "airport", "baseballfield", 
               "basketballcourt", "bridge", "chimney", 
               "expressway-service-area", "expressway-toll-station", "dam",
               "golffield", "groundtrackfield", "harbor",
               "overpass", "ship", "stadium",
               "storagetank", "tenniscourt", "trainstation",
               "vehicle", "windmill"
               )
    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):

        super(DiorDataset, self).__init__(ann_file,
                                           pipeline,
                                           data_root,
                                           img_prefix,
                                           seg_prefix,
                                           proposal_file,
                                           test_mode,
                                           filter_empty_gt)
    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            bbox= ann['bbox']
            
            [x1, y1, x2, y2, x3, y3, x4, y4] = bbox
            x_min = min(x1, x2, x3, x4)
            x_max = max(x1, x2, x3, x4)
            y_min = min(y1, y2, y3, y4)
            y_max = max(y1, y2, y3, y4)
            w = x_max - x_min
            h = y_max - y_min

            if ann['area'] <= 16 or w < 1 or h < 1:
                continue

            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 8), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 8), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann
    
