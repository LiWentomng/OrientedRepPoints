# -*- coding: gbk -*-
import numpy as np
from ..registry import PIPELINES
import cv2
import mmcv
import random 
from mmdet.core import poly2rbox, rbox2poly
'''
images and oriented bounding boxes transformation functions. poly boxes: (n, 8)
'''

@PIPELINES.register_module
class CorrectOBBox(object):
    """
    Correct gt_oboxes, turn gt_oboxes(n, 8) to rotate rectangle(n, 8).

    Args:
        correct_obbox (bool): Whether to shape the gt_oboxes(n, 8) to be rotate rectangle(n, 8).
        refine_obbox(bool):  Whether to keep the original points order.
    """
    def __init__(self, correct_obbox=True, refine_obbox=False):
        self.correct_obbox = correct_obbox
        self.refine_obbox = refine_obbox

    def _correct_obbox(self, gt_obboxes_points, refine_obbox=False):
        gt_obboxes_points_correct = []
        for obbox_points in gt_obboxes_points:
            obbox_points_4x2 = obbox_points.astype(np.int64).reshape(4, 2)
            rbbox_xywht = cv2.minAreaRect(obbox_points_4x2)
            x_ctr, y_ctr, width, height, theta = rbbox_xywht[0][0], rbbox_xywht[0][1], \
                                                 rbbox_xywht[1][0], rbbox_xywht[1][1], rbbox_xywht[2]
            obbox_points = cv2.boxPoints(((x_ctr, y_ctr), (width, height), theta)).reshape(-1)
            if refine_obbox:
                min_dist = 1e8
                for i, rbbox_point in enumerate(obbox_points.reshape(4, 2)):
                    ori_x1, ori_y1 = obbox_points_4x2[0]
                    cur_x1, cur_y1 = rbbox_point
                    dist = np.sqrt((ori_x1 - cur_x1) ** 2 + (ori_y1 - cur_y1) ** 2)
                    if dist <= min_dist:
                        min_dist = dist
                        index = i
                gt_obboxes_correct = np.array([
                    obbox_points[2 * (index % 4)], obbox_points[2 * (index % 4) + 1],
                    obbox_points[2 * ((index + 1) % 4)], obbox_points[2 * ((index + 1) % 4) + 1],
                    obbox_points[2 * ((index + 2) % 4)], obbox_points[2 * ((index + 2) % 4) + 1],
                    obbox_points[2 * ((index + 3) % 4)], obbox_points[2 * ((index + 3) % 4) + 1],
                ])
                gt_obboxes_points_correct.append(gt_obboxes_correct)
            else:
                gt_obboxes_points_correct.append(obbox_points)

        return np.array(gt_obboxes_points_correct)

    def __call__(self, results):
        if self.correct_obbox:
            gt_obboxes_points = results['gt_bboxes']
            gt_obboxes_points_correct = self._correct_obbox(gt_obboxes_points, self.refine_obbox)
            results['gt_bboxes'] = gt_obboxes_points_correct.astype(np.float32)
        return results


@PIPELINES.register_module
class PolyResize(object):
    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 clamp_rbbox=True,  # False
                 interpolation='bilinear'):
        """
        Resize poly format labels(n, 8) and images.
        Args:
            img_scale (tuple or list[tuple]): Images scales for resizing.
            multiscale_mode (str): Either "range" or "value".
            ratio_range (tuple[float]): (min_ratio, max_ratio)
            keep_ratio (bool): Whether to keep the aspect ratio when resizing the
                image. Defaults to True.
            clamp_rbbox(bool, optional): Whether clip the objects outside
                the border of the image. Defaults to True.
            interpolation: Interpolation method, accepted values are
                "nearest", "bilinear", "bicubic", "area", "lanczos".
        """
        self.clamp_rbbox = clamp_rbbox
        self.interpolation = interpolation
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):  # img_scale=[(1333, 768), (1333, 1280)]
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)
        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:  # img_scale=[(1333, 768), (1333, 1280)]
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)  # return img_scale = (long_edge, short_edge), None
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale  # results['scale'] = (long_edge, short_edge)
        results['scale_idx'] = scale_idx  # results['scale_idx'] = None

    def _resize_img(self, results):
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True, interpolation=self.interpolation)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True, interpolation=self.interpolation)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results, clamp_rbbox=True):
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            if clamp_rbbox:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            results[key] = bboxes

    def normal_call(self, results):
        if 'scale' not in results:
            self._random_scale(results)
        else:  # test_aug
            assert len(results['scale']) == 2
            edge1 = np.random.randint(
                min(results['scale']),
                max(results['scale']) + 1)
            edge2 = np.random.randint(
                min(results['scale']),
                max(results['scale']) + 1)
            results['scale'] = (max(edge1, edge2) + 1, min(edge1, edge2))
        self._resize_img(results)
        self._resize_bboxes(results, self.clamp_rbbox)

        return results

    def multi_img_call(self, results_4or9):
        for results in results_4or9:
            if 'scale' not in results:
                self._random_scale(results)
            self._resize_img(results)
            self._resize_bboxes(results, self.clamp_rbbox)  #

        return results_4or9

    def __call__(self, results):
        if not isinstance(results, list):
            results = self.normal_call(results)
        else:
            results = self.multi_img_call(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(img_scale={}, multiscale_mode={}, ratio_range={}, '
                     'keep_ratio={}, clamp_rbbox={}, interpolation={})').format(self.img_scale,
                                                                                self.multiscale_mode,
                                                                                self.ratio_range,
                                                                                self.keep_ratio,
                                                                                self.clamp_rbbox,
                                                                                self.interpolation)
        return repr_str



@PIPELINES.register_module
class PolyRandomFlip(object):
    """Flip the image & obbox(n, 8) (poly box)
    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.
    Args:
        flip_ratio (float, optional): The flipping probability.
            Default: None.
        direction (list[str]): The flipping direction. Options
            are 'horizontal', 'vertical'.
    """

    def __init__(self, flip_ratio=None, direction=['horizontal', 'vertical']):
        self.flip_ratio = flip_ratio
        self.direction = direction
       #  assert isinstance(direction, list)
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1
        for d in self.direction:
            assert d in ['horizontal', 'vertical']
       #  assert direction in ['horizontal', 'vertical']

    def rbbox_flip(self, rbboxes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            rbboxes(ndarray): shape (..., 8*k)
            img_shape(tuple): (height, width)
        """
        assert rbboxes.shape[-1] % 8 == 0
        flipped = rbboxes.copy()
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
    
    def normal_call(self, results):
        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        #if 'flip_direction' not in results:
        results['flip_direction'] = random.sample(self.direction, 1)[0]
       #      results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(
                results['img'], direction=results['flip_direction'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.rbbox_flip(results[key],
                                              results['img_shape'],
                                              results['flip_direction'])
        return results

    def multi_img_call(self, results_4or9):
        for results in results_4or9:
            if 'flip' not in results:
                flip = True if np.random.rand() < self.flip_ratio else False
                results['flip'] = flip
            #if 'flip_direction' not in results:
            results['flip_direction'] = random.sample(self.direction, 1)[0]
           #      results['flip_direction'] = self.direction
            if results['flip']:
                # flip image
                results['img'] = mmcv.imflip(
                    results['img'], direction=results['flip_direction'])
                # flip bboxes
                for key in results.get('bbox_fields', []):
                    results[key] = self.rbbox_flip(results[key],
                                                results['img_shape'],
                                                results['flip_direction'])
        return results_4or9

    def __call__(self, results):
        if not isinstance(results, list):
            results = self.normal_call(results)
        else:
            results = self.multi_img_call(results)
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={}, direction={})'.format(
            self.flip_ratio, self.direction)


@PIPELINES.register_module
class PolyRandomRotate(object):
    """
    Rotate img & poly bbox(n, 8).
    Args:
        rate (bool): (float, optional): The rotating probability.
            Default: 0.5.
        angles_range(int): The rotate angle defined by random(-angles_range, +angles_range).
        auto_bound(bool): whether to find the new width and height bounds.
    """
    def __init__(self,
                 rotate_ratio=0.5,
                 angles_range=180,    # random(-180, 180)
                 auto_bound=False):
        self.rotate_ratio = rotate_ratio
        self.auto_bound = auto_bound
        self.angles_range = angles_range
        self.discrete_range = [90, 180, -90, -180]

    @property
    def is_rotate(self):
        return np.random.rand() < self.rotate_ratio

    def apply_image(self, img, bound_h, bound_w, interp=cv2.INTER_LINEAR):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if len(img) == 0:
            return img
        return cv2.warpAffine(img, self.rm_image, (bound_w, bound_h), flags=interp)

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y) points
        """
        if len(coords) == 0:
            return coords
        coords = np.asarray(coords, dtype=float)
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=cv2.INTER_NEAREST)
        return segmentation

    def create_rotation_matrix(self, center, angle, bound_h, bound_w, offset=0):
        center = (center[0] + offset, center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), angle, 1)
        if self.auto_bound:
            rot_im_center = cv2.transform(
                center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array(
                [bound_w / 2, bound_h / 2]) + offset - rot_im_center
            # shift the rotation center to the new coordinates
            rm[:, 2] += new_center
        return rm

    def filter_border(self, bboxes, h, w): # bboxes: (n, 5)
        x_ctr, y_ctr = bboxes[:, 0], bboxes[:, 1]
        w_bbox, h_bbox = bboxes[:, 2], bboxes[:, 3]
        keep_inds = (x_ctr > 0) & (x_ctr < w) & (y_ctr > 0) & (y_ctr < h) & (w_bbox > 5) & (h_bbox > 5)
        return keep_inds
    
    def normal_call(self, results):
        # return the results directly if not rotate
        if not self.is_rotate:
            results['rotate'] = False
            angle = 0
        else:
            angle = random.uniform( -self.angles_range, self.angles_range)
            results['rotate'] = True
            # 'storage-tank' 'roundabout'
            # class_labels = results['gt_labels'] # (n)
            # for classid in class_labels:
            #     if classid == 10 or classid == 12:
            #         random.shuffle(self.discrete_range)
            #         angle = self.discrete_range[0]
            #         break

        h, w, c = results['img_shape']
        img = results['img']
        # angle for rotate
        # angle = random.uniform( -self.angles_range, self.angles_range)
        # results['rotate'] = True
        results['rotate_angle'] = angle

        image_center = np.array((w / 2, h / 2))
        abs_cos, abs_sin = abs(np.cos(angle)), abs(np.sin(angle))
        if self.auto_bound:
            # find the new width and height bounds
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
            ).astype(int)
        else:
            bound_w, bound_h = w, h

        self.rm_coords = self.create_rotation_matrix(
            image_center, angle, bound_h, bound_w)
        # Needed because of this problem https://github.com/opencv/opencv/issues/11784
        self.rm_image = self.create_rotation_matrix(
            image_center, angle, bound_h, bound_w, offset=-0.5)
        # rotate img
        img = self.apply_image(img, bound_h, bound_w)
        results['img'] = img
        results['img_shape'] = (bound_h, bound_w, c)
        gt_bboxes = results.get('gt_bboxes', [])
        labels = results.get('gt_labels', [])

        polys = gt_bboxes.reshape(-1,2)
        polys = self.apply_coords(polys).reshape(-1, 8)
        gt_bboxes = poly2rbox(polys)
        keep_inds = self.filter_border(gt_bboxes, bound_h, bound_w)
        gt_bboxes = gt_bboxes[keep_inds, :]
        labels = labels[keep_inds]
        if len(gt_bboxes) == 0:
            return None
        results['gt_bboxes'] = rbox2poly(gt_bboxes).astype(np.float32)
        results['gt_labels'] = labels

        return results
    
    def multi_img_call(self, results_4or9):
        for results in results_4or9:
            # return the results directly if not rotate
            if not self.is_rotate:
                results['rotate'] = False
                angle = 0
            else:
                angle = random.uniform( -self.angles_range, self.angles_range)
                results['rotate'] = True
                # storage-tank' 'roundabout'  90
                # class_labels = results['gt_labels'] # (n)
                # for classid in class_labels:
                #     if classid == 10 or classid == 12:
                #         random.shuffle(self.discrete_range)
                #         angle = self.discrete_range[0]
                #         break

            h, w, c = results['img_shape']
            img = results['img']
            # angle for rotate
            #angle = self.rand_angle
            # angle = random.uniform( -self.angles_range, self.angles_range)
            # results['rotate'] = True
            results['rotate_angle'] = angle
            image_center = np.array((w / 2, h / 2))
            abs_cos, abs_sin = abs(np.cos(angle)), abs(np.sin(angle))
            if self.auto_bound:
                # find the new width and height bounds
                bound_w, bound_h = np.rint(
                    [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
                ).astype(int)
            else:
                bound_w, bound_h = w, h

            self.rm_coords = self.create_rotation_matrix(
                image_center, angle, bound_h, bound_w)
            # Needed because of this problem https://github.com/opencv/opencv/issues/11784
            self.rm_image = self.create_rotation_matrix(
                image_center, angle, bound_h, bound_w, offset=-0.5)
            # rotate img
            img = self.apply_image(img, bound_h, bound_w)
            results['img'] = img
            results['img_shape'] = (bound_h, bound_w, c)
            gt_bboxes = results.get('gt_bboxes', [])
            labels = results.get('gt_labels', [])

            polys = gt_bboxes.reshape(-1,2)
            polys = self.apply_coords(polys).reshape(-1, 8)
            gt_bboxes = poly2rbox(polys)
            keep_inds = self.filter_border(gt_bboxes, bound_h, bound_w)
            gt_bboxes = gt_bboxes[keep_inds, :]
            labels = labels[keep_inds]
            if len(gt_bboxes) == 0:
                results = None
                continue
            results['gt_bboxes'] = rbox2poly(gt_bboxes).astype(np.float32)
            results['gt_labels'] = labels

        return results_4or9

    def __call__(self, results):
        if not isinstance(results, list):
            results = self.normal_call(results)
        else:
            results = self.multi_img_call(results)
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(rotate_ratio={}, angles_range={}, auto_bound={})').format(self.rotate_ratio,
                                                                         self.angles_range,
                                                                         self.auto_bound)
        return repr_str

