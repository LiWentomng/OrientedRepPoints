# Modified from https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/cityscapes.py # noqa
# and https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa

import glob
import os
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.utils import print_log
from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class CityscapesDataset(CocoDataset):

    CLASSES = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            img_id = img_info['id']
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann_info = self.coco.loadAnns(ann_ids)
            all_iscrowd = all([_['iscrowd'] for _ in ann_info])
            if self.filter_empty_gt and (self.img_ids[i] not in ids_with_ann
                                         or all_iscrowd):
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            img_info (dict): Image info of an image.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map.
                "masks" are already decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
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
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=img_info['segm_file'])

        return ann

    def results2txt(self, results, outfile_prefix):
        """Dump the detection results to a txt file.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files.
                If the prefix is "somepath/xxx",
                the txt files will be named "somepath/xxx.txt".

        Returns:
            list[str: str]: result txt files which contains corresponding
            instance segmentation images.
        """
        try:
            import cityscapesscripts.helpers.labels as CSLabels
        except ImportError:
            raise ImportError('Please run "pip install citscapesscripts" to '
                              'install cityscapesscripts first.')
        result_files = []
        os.makedirs(outfile_prefix, exist_ok=True)
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]
            pred_txt = osp.join(outfile_prefix, basename + '_pred.txt')

            bbox_result, segm_result = result
            bboxes = np.vstack(bbox_result)
            segms = mmcv.concat_list(segm_result)
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)

            assert len(bboxes) == len(segms) == len(labels)
            num_instances = len(bboxes)
            prog_bar.update()
            with open(pred_txt, 'w') as fout:
                for i in range(num_instances):
                    pred_class = labels[i]
                    classes = self.CLASSES[pred_class]
                    class_id = CSLabels.name2label[classes].id
                    score = bboxes[i, -1]
                    mask = maskUtils.decode(segms[i]).astype(np.uint8)
                    png_filename = osp.join(
                        outfile_prefix,
                        basename + '_{}_{}.png'.format(i, classes))
                    mmcv.imwrite(mask, png_filename)
                    fout.write('{} {} {}\n'.format(
                        osp.basename(png_filename), class_id, score))
            result_files.append(pred_txt)

        return result_files

    def format_results(self, results, txtfile_prefix=None):
        """Format the results to txt (standard format for Cityscapes evaluation).

        Args:
            results (list): Testing results of the dataset.
            txtfile_prefix (str | None): The prefix of txt files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving txt/png files when txtfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if txtfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            txtfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2txt(results, txtfile_prefix)

        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 outfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        """Evaluation in Cityscapes protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            outfile_prefix (str | None):
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str: float]
        """
        eval_results = dict()

        metrics = metric.copy() if isinstance(metric, list) else [metric]

        if 'cityscapes' in metrics:
            eval_results.update(
                self._evaluate_cityscapes(results, outfile_prefix, logger))
            metrics.remove('cityscapes')

        # left metrics are all coco metric
        if len(metrics) > 0:
            # create CocoDataset with CityscapesDataset annotation
            self_coco = CocoDataset(self.ann_file, self.pipeline.transforms,
                                    self.data_root, self.img_prefix,
                                    self.seg_prefix, self.proposal_file,
                                    self.test_mode, self.filter_empty_gt)
            eval_results.update(
                self_coco.evaluate(results, metrics, logger, outfile_prefix,
                                   classwise, proposal_nums, iou_thrs))

        return eval_results

    def _evaluate_cityscapes(self, results, txtfile_prefix, logger):
        try:
            import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as CSEval  # noqa
        except ImportError:
            raise ImportError('Please run "pip install citscapesscripts" to '
                              'install cityscapesscripts first.')
        msg = 'Evaluating in Cityscapes style'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        result_files, tmp_dir = self.format_results(results, txtfile_prefix)

        if tmp_dir is None:
            result_dir = osp.join(txtfile_prefix, 'results')
        else:
            result_dir = osp.join(tmp_dir.name, 'results')

        eval_results = {}
        print_log(
            'Evaluating results under {} ...'.format(result_dir),
            logger=logger)

        # set global states in cityscapes evaluation API
        CSEval.args.cityscapesPath = os.path.join(self.img_prefix, '../..')
        CSEval.args.predictionPath = os.path.abspath(result_dir)
        CSEval.args.predictionWalk = None
        CSEval.args.JSONOutput = False
        CSEval.args.colorized = False
        CSEval.args.gtInstancesFile = os.path.join(result_dir,
                                                   'gtInstances.json')
        CSEval.args.groundTruthSearch = os.path.join(
            self.img_prefix.replace('leftImg8bit', 'gtFine'),
            '*/*_gtFine_instanceIds.png')

        groundTruthImgList = glob.glob(CSEval.args.groundTruthSearch)
        assert len(groundTruthImgList), \
            'Cannot find ground truth images in {}.'.format(
                CSEval.args.groundTruthSearch)
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(CSEval.getPrediction(gt, CSEval.args))
        CSEval_results = CSEval.evaluateImgLists(predictionImgList,
                                                 groundTruthImgList,
                                                 CSEval.args)['averages']

        eval_results['mAP'] = CSEval_results['allAp']
        eval_results['AP@50'] = CSEval_results['allAp50%']
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
