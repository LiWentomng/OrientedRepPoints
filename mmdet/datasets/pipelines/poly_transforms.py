# -*- coding: gbk -*-
import numpy as np
from ..registry import PIPELINES
import cv2
import mmcv
import random 
from mmdet.core import poly2rbox, rbox2poly
import math
from collections import Counter
import matplotlib.pyplot as plt
import copy
plt.set_loglevel('WARNING')

@PIPELINES.register_module
class CorrectRBBox(object):
    """
    Correct gt_bboxes, turn gt_bboxes(n, 8) to rotate rectangle(n, 8).

    Args:
        correct_rbbox (bool): Whether to shape the gt_bboxes(n, 8) to be rotate rectangle(n, 8).
        refine_rbbox(bool):  Whether to keep the original points order.
    """
    def __init__(self, correct_rbbox=True, refine_rbbox=False):
        self.correct_rbbox = correct_rbbox
        self.refine_rbbox = refine_rbbox
    # correct_rbbox: 是否将任意四点坐标correct成最小外界矩形  refine_rbbox: 是否更改四点坐标的先后顺序
    def _correct_rbbox(self, gt_rbboxes_points, refine_rbbox=False): # gt_rbboxes_points:(n, 8)
        gt_bboxes_points_correct = [] 
        for rbbox_points in gt_rbboxes_points:  # rbbox_points：array.shape(8)
            rbbox_points_4x2 = rbbox_points.astype(np.int64).reshape(4, 2)
            rbbox_xywht = cv2.minAreaRect(rbbox_points_4x2)
            x_ctr, y_ctr, width, height, theta = rbbox_xywht[0][0], rbbox_xywht[0][1], \
                                                 rbbox_xywht[1][0], rbbox_xywht[1][1], rbbox_xywht[2]
            rbbox_points = cv2.boxPoints(((x_ctr, y_ctr), (width, height), theta)).reshape(-1)  # rbbox_points:(8)
            if refine_rbbox:
                min_dist = 1e8
                for i, rbbox_point in enumerate(rbbox_points.reshape(4, 2)):
                    ori_x1, ori_y1 = rbbox_points_4x2[0]
                    cur_x1, cur_y1 = rbbox_point
                    dist = np.sqrt((ori_x1 - cur_x1) ** 2 + (ori_y1 - cur_y1) ** 2)
                    if dist <= min_dist:
                        min_dist = dist
                        index = i
                gt_bboxes_correct = np.array([   # gt_bboxes_correct： array.shape(8)
                    rbbox_points[2 * (index % 4)], rbbox_points[2 * (index % 4) + 1],
                    rbbox_points[2 * ((index + 1) % 4)], rbbox_points[2 * ((index + 1) % 4) + 1],
                    rbbox_points[2 * ((index + 2) % 4)], rbbox_points[2 * ((index + 2) % 4) + 1],
                    rbbox_points[2 * ((index + 3) % 4)], rbbox_points[2 * ((index + 3) % 4) + 1],
                ])
                gt_bboxes_points_correct.append(gt_bboxes_correct)
            else:
                gt_bboxes_points_correct.append(rbbox_points)

        return np.array(gt_bboxes_points_correct) # return array.shape(n, 8)
    
    def normal_call(self, results):
        gt_rbboxes_points = results['gt_bboxes']  # results['gt_bboxes'] (n, 8)
        gt_rbboxes_points_correct = self._correct_rbbox(gt_rbboxes_points, self.refine_rbbox) # gt_rbboxes_points_correct: array.shape(n, 8)
        results['gt_bboxes'] = gt_rbboxes_points_correct.astype(np.float32)
        
        return results
    
    def multi_img_call(self, results_4or9):
        for results in results_4or9:
            gt_rbboxes_points = results['gt_bboxes']  # results['gt_bboxes'] (n, 8)
            gt_rbboxes_points_correct = self._correct_rbbox(gt_rbboxes_points, self.refine_rbbox) # gt_rbboxes_points_correct: array.shape(n, 8)
            results['gt_bboxes'] = gt_rbboxes_points_correct.astype(np.float32)
        
        return results_4or9

    def __call__(self, results):
        if self.correct_rbbox:
            if not isinstance(results, list):
                results = self.normal_call(results)
            else:
                results = self.multi_img_call(results)
        return results
    
    def __repr__(self):  # 实例化对象时，可以获得自我描述信息
        repr_str = self.__class__.__name__
        repr_str += ('(correct_rbbox={}, refine_rbbox={})').format(self.correct_rbbox,
                                              self.refine_rbbox)
        return repr_str

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
            if isinstance(img_scale, list):   # img_scale=[(1333, 768), (1333, 1280)]
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
    def random_sample(img_scales):  # img_scale=[(1333, 768), (1333, 1280)]
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]  # 最大边 [max, max]
        # print('##############')
        # print('img_scale_long', img_scale_long)
        img_scale_short = [min(s) for s in img_scales] # 最小边 [min, min]
        # print('img_scale_short', img_scale_short)
        long_edge = np.random.randint( 
            min(img_scale_long),
            max(img_scale_long) + 1)
        # print('long_edge', long_edge)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        # print('short_edge', short_edge)
        img_scale = (long_edge, short_edge)
        # print('img_scale', img_scale)
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
        elif len(self.img_scale) == 1: # img_scale=[(1333, 768), (1333, 1280)]
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)  # return img_scale = (long_edge, short_edge), None
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale   # results['scale'] = (long_edge, short_edge)
        results['scale_idx'] = scale_idx # results['scale_idx'] = None

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
            self._random_scale(results)   # 给results['scale_idx'] 和 results['scale']赋值
        else:  # test_aug
            assert len(results['scale']) == 2  # (2048, 1333)
            edge1 = np.random.randint( 
                min(results['scale']),
                max(results['scale']) + 1)
            edge2 = np.random.randint(
                min(results['scale']),
                max(results['scale']) + 1)
            results['scale'] = (max(edge1, edge2)+1, min(edge1, edge2))
        self._resize_img(results)
        self._resize_bboxes(results, self.clamp_rbbox)

        return results

    def multi_img_call(self, results_4or9):
        for results in results_4or9:
            if 'scale' not in results:
                self._random_scale(results)   # 随机采样scale并给results['scale_idx'] 和 results['scale']赋值
            self._resize_img(results)         # 等ratio的随机比例缩放
            self._resize_bboxes(results, self.clamp_rbbox) #
        
        return results_4or9

    def __call__(self, results):
        if not isinstance(results, list):
            results = self.normal_call(results)
        else:
            results = self.multi_img_call(results)
        return results

    def __repr__(self):  # 实例化对象时，可以获得自我描述信息
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
    """Flip the image & bbox(n, 8)

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
    Rotate img & bbox(n, 8).

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
        self.discrete_range = [90, 180, -90, -180]  # 水平框物体时的旋转角度

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
            # Find the coordinates of the center of rotation in the new image
            # The only point for which we know the future coordinates is the center of the image
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

            # 图中物体类别为'storage-tank' 'roundabout' 'airport'时只进行90°的旋转
            # class_labels = results['gt_labels'] # (n)
            # for classid in class_labels:
            #     if classid == 10 or classid == 12 or classid ==17: # class_num=18时
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

                # 图中物体类别为'storage-tank' 'roundabout' 'airport'时只进行90°的旋转
                # class_labels = results['gt_labels'] # (n)
                # for classid in class_labels:
                #     if classid == 12 or classid == 16 or classid ==17:  # class_num=18时
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
    
    def __repr__(self):  # 实例化对象时，可以获得自我描述信息
        repr_str = self.__class__.__name__
        repr_str += ('(rotate_ratio={}, angles_range={}, auto_bound={})').format(self.rotate_ratio,
                                                                         self.angles_range,
                                                                         self.auto_bound)
        return repr_str

@PIPELINES.register_module
class Poly_Mosaic_RandomPerspective(object):
    """
    Mosaic augmentation.

    Given 4 or 9 images, mosaic combine them into one output image
                
              output Mosaic4_mode image
                             cut_x
            +------------------------------+
            |                 |            |
            |    image 0      |  image 1   |
      cut_y |-----------------|------------|
            |                 |            |
            |    image 2      |  image3    |
            |                 |            |
            |                 |            |
            +------------------------------|


              output Mosaic9_mode image
        +-------------------------------------------+
        |           |         |                     |
        |           | image 1 | image 2             |
        |   image 8 |         |                     |
        |           |---------------------|---------|      
        |           |                     |         |
        |-----------|                     | image 3 |
        |           |      image 0        |         |
        |   image 7 |                     |---------|
        |           |                     |         |
        |-----------|---------------------| image 4 |
        |               |                 |         |
        |   image 6     |   image 5       |         |
        |               |                 |         |
        +-------------------------------------------+

    Args:
        degrees(int) : The rotate augmentation after mosaic, the rotate angle defined by random.uniform(-degrees, degrees).
            Default: 0.
        translate(int) : The translate augmentation after mosaic.
            Default: 0.
        scale(int) : Resize mosaic to random(1-scale, 1+scale) size-ratio.
            Default: 0.
        shear(int) : The shear augmentation after mosaic, the shear angle(°) defined by random.uniform(-degrees, degrees).
            Default: 0.
        perspective(float) : The perspective augmentation after mosaic.
            Default: 0.
        Mosaic_Crop(bool) : Whether to crop mosaic, the size of crop output is defined by the max size of inputs.
            Default: True
        rate: The mosaic implement probability.
            Default: 0.5

    About output size:
            Given 4 images, which sizes are (1024, 1024), (1280, 1280), (1536, 1536), (768, 768).
            if Mosaic4_mode and not Mosaic_Crop:
                The output size is (3072, 3072)
            if Mosaic9_mode and not Mosaic_Crop:
                The output size is (4608, 4608)
            if Mosaic?_mode and Mosaic_Crop:
                The output size is (1536, 1536)
            if Mixup_mode:
                The output is List[mosaic_output1, mosaic_output2]
    """

    def __init__(self,
                 degrees=0,
                 translate=0,
                 scale=0,
                 shear=0, 
                 perspective=0.0,
                 ifcrop=True,
                 mosaic_ratio=0.5
                 ): 
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.random_perspective_flag = ifcrop
        self.mosaic_ratio = mosaic_ratio
    
    def load_mosaic4(self, results_4):
        labels4 = []
        gt_bboxes4 = []
        s = self.img_size
        # 随机取mosaic中心点
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y

        for i, results in enumerate(results_4):

            # Load image
            # img.size = (height, width, 3)
            img = results['img']
            h, w = img.shape[0], img.shape[1]

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles  img4.shape(2048, 2048, 3)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)  用于确定原图片在img4左上角的坐标（左上右下）
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)  用于确定原图片剪裁进img4中的图像内容范围
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            # img4.size = [resized_height,resized_ width, 3]
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b   # 原图片未剪裁进img4中的宽度
            padh = y1a - y1b   # 原图片未剪裁进img4中的高度

            # Labels
            x = results['gt_bboxes']  # x.shape(n, [x1 y1 x2 y2 x3 y3 x4 y4]) 
            labels = results['gt_labels'] # labels.shape (n)
            gt_bboxes = x.copy()
            if x.size > 0:  
                gt_bboxes[:, 0::2] = x[:, 0::2] + padw
                gt_bboxes[:, 1::2] = x[:, 1::2] + padh
            gt_bboxes4.append(gt_bboxes)  # labels4：[array.size(n1, 8), array.size(n2, 8), array.size(n3, 8), array.size(n4, 8)]
            labels4.append(labels)
        
        # Concat/clip labels
        if len(gt_bboxes4):
            # 即labels4.shape=(一张mosaic图片中的GT数量, [x1 y1 x2 y2 x3 y3 x4 y4])
            gt_bboxes4 = np.concatenate(gt_bboxes4, 0)  # 将第一个维度取消
            labels4 = np.concatenate(labels4, 0)
            np.clip(gt_bboxes4[:, :], 0, 2 * s, out=gt_bboxes4[:, :])  # 限定labels4[:, :]中最小值只能为0，最大值只能为2*self.size
        
        return img4, gt_bboxes4, labels4
    
    def load_mosaic9(self, results_9):
        labels9 = []
        gt_bboxes9 = []
        s = self.img_size

        for i, results in enumerate(results_9):
            # Load image
            # img.size = (height, width, 3)
            img = results['img']
            h, w = img.shape[0], img.shape[1]

            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords
                
            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

            # Labels
            x = results['gt_bboxes']  # x.shape(n, [x1 y1 x2 y2 x3 y3 x4 y4]) 
            labels = results['gt_labels'] # labels.shape (n)
            gt_bboxes = x.copy()
            if x.size > 0:  
                gt_bboxes[:, 0::2] = x[:, 0::2] + padx
                gt_bboxes[:, 1::2] = x[:, 1::2] + pady
            gt_bboxes9.append(gt_bboxes)  # gt_bboxes9 ：[array.size(n1, 8), array.size(n2, 8), array.size(n3, 8), array.size(n4, 8)]
            labels9.append(labels)
        
        # # Offset
        yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]  # 截取2s * 2s的图像区域
        
        # Concat/clip labels
        if len(gt_bboxes9):
            # 即labels4.shape=(一张mosaic图片中的GT数量, [x1 y1 x2 y2 x3 y3 x4 y4])
            gt_bboxes9 = np.concatenate(gt_bboxes9, 0)  # 将第一个维度取消
            labels9 = np.concatenate(labels9, 0)
            gt_bboxes9[:, 0::2] -= xc
            gt_bboxes9[:, 1::2] -= yc
            np.clip(gt_bboxes9[:, :], 0, 2 * s, out=gt_bboxes9[:, :])  # 限定labels9[:, :]中最小值只能为0，最大值只能为2*self.size
        
        return img9, gt_bboxes9, labels9

    def random_perspective(self, img, bboxes=(), labels=(), degrees=0, translate=0, scale=0, shear=0, perspective=0.0, border=(0, 0)):
        '''
        遍性数据增强：
                进行随机旋转，缩放，错切，平移，center，perspective数据增强
        Args:
            img: shape=(height_mosaic, width_mosaic, 3)
            targets ：size = (n, 8) 未归一化  （归一化的数据无法处理）
        Returns:
            img：shape=(height, width, 3)
            targets = (n, 8)
        '''

        height = img.shape[0] + border[0] * 2  # shape(h,w,c) 用于将Mosaic图像剪裁至要求的大小 相当于2*img_size - img_size
        width = img.shape[1] + border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # 设置旋转和缩放的仿射矩阵并进行旋转和缩放
        # Rotation and Scale
        R = np.eye(3)  # 行数为3,对角线为1,其余为0的矩阵
        a = random.uniform(-degrees, degrees)   # 随机生成[-degrees, degrees)的实数 即为旋转角度 负数则代表逆时针旋转
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - scale, 1 + scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)  # 获得以(0,0)为中心的旋转仿射变化矩阵

        # 设置裁剪的仿射矩阵系数
        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # 设置平移的仿射系数
        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

        # Combined rotation matrix
        # 融合仿射矩阵并作用在图片上
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        # Transform label coordinates
        # 调整框的标签
        n = len(bboxes)  # targets.size = (n, 8)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = bboxes[:, :].reshape(n * 4, 2)  # x1y1, ,x2y2 , x3y3, x4y4
            xy = xy @ M.T  # transform
            if perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            else:  # affine
                xy = xy[:, :2].reshape(n, 8)

            # # clip boxes
            # xy_ = xy.copy()  # (n, 8)
            # xy_[:, [0, 2, 4, 6]] = xy[:, [0, 2, 4, 6]].clip(0, width)
            # xy_[:, [1, 3, 5, 7]] = xy[:, [1, 3, 5, 7]].clip(0, height)

            # filter candidates
            rbboxes = poly2rbox(xy)  # (n,5)
            keep_inds = self.filter_border(rbboxes, height, width)
            xy = xy[keep_inds, :]

            bboxes = xy
            labels = labels[keep_inds]

        return img, bboxes, labels

    def filter_border(self, bboxes, h, w): # bboxes.size(n,5)
        x_ctr, y_ctr = bboxes[:, 0], bboxes[:, 1]
        w_bboxes, h_bboxes = bboxes[:, 2], bboxes[:, 3]
        keep_inds = (x_ctr > 0) & (x_ctr < w) & (y_ctr > 0) & (y_ctr < h) & (w_bboxes > 5) & (h_bboxes > 5)
        return keep_inds

    def normal_call(self, results):
        return results
    
    def mosaic4_call(self, results_4):
        img_mosaic4, gt_bboxes_mosaic4, gt_labels_mosaic4 = self.load_mosaic4(results_4)
        if self.random_perspective_flag:
            img_mosaic4, gt_bboxes_mosaic4, gt_labels_mosaic4= self.random_perspective(
                img=img_mosaic4,
                bboxes= gt_bboxes_mosaic4,
                labels=gt_labels_mosaic4,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=self.mosaic_border
            )
        else:
            # filter candidates
            rbboxes = poly2rbox(gt_bboxes_mosaic4)  # (n, 8) -> (n,5)
            keep_inds = self.filter_border(rbboxes, img_mosaic4.shape[0], img_mosaic4.shape[1])
            gt_bboxes_mosaic4 = gt_bboxes_mosaic4[keep_inds, :]
            gt_labels_mosaic4 = gt_labels_mosaic4[keep_inds]

        results = results_4[0]
        results['img'] = img_mosaic4
        results['gt_bboxes'] = gt_bboxes_mosaic4.astype(np.float32)
        results['gt_labels'] = gt_labels_mosaic4
        return results

    def mosaic9_call(self, results_9):
        img_mosaic9, gt_bboxes_mosaic9, gt_labels_mosaic9 = self.load_mosaic9(results_9)
        if self.random_perspective_flag:
            img_mosaic9, gt_bboxes_mosaic9, gt_labels_mosaic9= self.random_perspective(
                img=img_mosaic9,
                bboxes= gt_bboxes_mosaic9,
                labels=gt_labels_mosaic9,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=self.mosaic_border
            )
        else:
            # filter candidates
            rbboxes = poly2rbox(gt_bboxes_mosaic9)  # (n, 8) -> (n,5)
            keep_inds = self.filter_border(rbboxes, img_mosaic9.shape[0], img_mosaic9.shape[1])
            gt_bboxes_mosaic9 = gt_bboxes_mosaic9[keep_inds, :]
            gt_labels_mosaic9 = gt_labels_mosaic9[keep_inds]
        results = results_9[0]
        results['img'] = img_mosaic9
        results['gt_bboxes'] = gt_bboxes_mosaic9.astype(np.float32)
        results['gt_labels'] = gt_labels_mosaic9
        return results

    def mixup_mosaic(self, results_x2):
        if len(results_x2) == 2:  # Normal + Mixup
            return results_x2

        results_pre = []
        results_last = []
        results_mixups = []
        for i, results in enumerate(results_x2):
            if i < (len(results_x2) / 2):
                results_pre.append(results)
            else:
                results_last.append(results)
        if results_x2[0]['Mosaic_mode'] == 'Mosaic4': # Mosaic4 + Mixup
            results_mixup1 = self.mosaic4_call(results_pre)
            results_mixup2 = self.mosaic4_call(results_last)

            results_mixups.append(results_mixup1)
            results_mixups.append(results_mixup2)
        elif results_x2[0]['Mosaic_mode'] == 'Mosaic9':  # Mosaic9 + Mixup
            results_mixup1 = self.mosaic9_call(results_pre)
            results_mixup2 = self.mosaic9_call(results_last)
            
            results_mixups.append(results_mixup1)
            results_mixups.append(results_mixup2)
        return results_mixups



    def __call__(self, results):
        if not isinstance(results, list):  # 1 img
            results = self.normal_call(results)
            return results

        self.img_size = 0
        for result in results:
            # img.size = (height, width, 3)
            img = result['img']
            img_max_size = max(img.shape[0], img.shape[1])
            self.img_size = max(self.img_size, img_max_size)
        self.mosaic_border = [-self.img_size // 2, -self.img_size // 2]

        if results[0]['Mixup_mode']:  # mixup = True
            if random.random() > self.mosaic_ratio: # with no mosaic process
                results_ = []
                results1 = results[0]
                results2 = results[int(len(results) / 2)]
                results_.append(results1)
                results_.append(results2)
                return results_
            else:
                results = self.mixup_mosaic(results)
                return results
        else: # mixup = False
            if random.random() > self.mosaic_ratio: # with no mosaic process
                results1 = results[0]
                return results1
            else:
                if results[0]['Mosaic_mode'] == 'Mosaic4':
                    results = self.mosaic4_call(results)
                    return results
                if results[0]['Mosaic_mode'] == 'Mosaic9':
                    results = self.mosaic9_call(results)
                    return results
    
    def __repr__(self):  # 实例化对象时，可以获得自我描述信息
        repr_str = self.__class__.__name__
        repr_str += ('(degrees={}, translate={}, scale={}, shear={}'
                     'perspective={}, ifcrop={}, mosaic_ratio={})').format(self.degrees,
                                                                        self.translate,
                                                                        self.scale,
                                                                        self.shear,
                                                                        self.perspective,
                                                                        self.random_perspective_flag,
                                                                        self.mosaic_ratio)                                                                             
        return repr_str

@PIPELINES.register_module
class MixUp(object):
    """mix 2 imgs

    Args:
        rate(float): the mixup rate
    """
    def __init__(self,
                 mixup_ratio=0.5
                 ): 
        self.mixup_ratio = mixup_ratio
    
    def mixup_imgs(self, results2):     
        results_1 = results2[0]
        results_2 = results2[1]

        img1, gt_bboxes1, gt_labels1 = results_1['img'], results_1['gt_bboxes'], results_1['gt_labels']
        img2, gt_bboxes2, gt_labels2 = results_2['img'], results_2['gt_bboxes'], results_2['gt_labels']
        
        max_h, max_w = max(img2.shape[0], img1.shape[0]), max(img2.shape[1], img1.shape[1])
        img1 = mmcv.impad(img1, (max_h, max_w), 0)
        img2 = mmcv.impad(img2, (max_h, max_w), 0)

        r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
        img_mixed = img1 * r + img2 * (1 - r)
        gt_bboxes = np.concatenate((gt_bboxes1, gt_bboxes2), 0)
        gt_labels = np.concatenate((gt_labels1, gt_labels2), 0)
        
        results_1['img'] = img_mixed
        results_1['gt_bboxes'] = gt_bboxes
        results_1['gt_labels'] = gt_labels

        return results_1
    
    def __call__(self, results):
        # print('mixup_results', len(results))
        # # print(results[0], results[1], results[2])
        # print(results.keys())
        # print(isinstance(results, list))

        if not isinstance(results, list): # only 1 img
            # print('only one')
            return results
        if random.random() < self.mixup_ratio: # 2 img
            results = self.mixup_imgs(results)
            # print('mixed', len(results))
            return results
        else:
            # print('unmixed', len(results[0]))
            return results[0]
    
    def __repr__(self):  # 实例化对象时，可以获得自我描述信息
        repr_str = self.__class__.__name__
        repr_str += ('(mixup_ratio={})').format(self.mixup_ratio)                                                                             
        return repr_str

@PIPELINES.register_module
class PolyImgPlot(object):
    """visualize the poly-format img after augmentation.

    Args:
        img_save_path (str): where to save the visualized img.
    """
    def __init__(self, img_save_path='work_dirs/', save_img_num=4, class_num=18, thickness=2):
        self.img_aug_id = 0
        self.img_save_path = img_save_path
        self.save_img_num = save_img_num
        # 设置画框的颜色    colors = [[178, 63, 143], [25, 184, 176], [238, 152, 129],....,[235, 137, 120]]随机设置RGB颜色
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(class_num)]
        self.thickness = thickness
        self.dict_class_img_distribution = {}
        self.dict_class_num_distribution = {}
        self.img_num = 0

    def __call__(self, results):
        dict_label_thisimg = Counter(results['gt_labels'])
        for i in range(1, len(self.colors)+1): # 1 ~ classnum
            if i in dict_label_thisimg:  # 该类别id在本次迭代中中出现
                # 对应类别id对应的键值自增，表示该类别存在的图片数量
                if i not in self.dict_class_img_distribution:
                    self.dict_class_img_distribution[i] = 1
                else:
                    self.dict_class_img_distribution[i] += 1
                # 对应类别id对应的键值 + 本次迭代的频数，表示该目标类别的数量
                if i not in self.dict_class_num_distribution:
                    self.dict_class_num_distribution[i] = dict_label_thisimg[i]
                else:
                    self.dict_class_num_distribution[i] += dict_label_thisimg[i]

        # if (results['Mosaic_mode'] != 'Normal') and (results['Mixup_mode'] == True):
        #     class_distribution_name = 'mixup+mosaic_mode_class_distribution.jpg'
        #     objects_distribution_name = 'mixup+mosaic_mode_objects_distribution.jpg'
        # elif results['Mixup_mode'] == True:
        #     class_distribution_name = 'mixup_mode_class_distribution.jpg'
        #     objects_distribution_name = 'mixup_mode_objects_distribution.jpg'
        # elif results['Mosaic_mode'] != 'Normal':
        #     class_distribution_name = 'mosaic_mode_class_distribution.jpg'
        #     objects_distribution_name = 'mosaic_mode_objects_distribution.jpg'
        # else:
        class_distribution_name = 'normal_mode_class_distribution.jpg'
        objects_distribution_name = 'normal_mode_objects_distribution.jpg'

        # 各个类别的分布图绘制
        plt_x = []
        plt_y = []
        self.img_num += 1
        for i in range(1, len(self.colors)+1):  #i: 1 ~ classnum
            if i in self.dict_class_img_distribution:
                plt_x.append('%g' % i)
                plt_y.append(self.dict_class_img_distribution[i] / self.img_num)
        fig = plt.figure(0)
        plt.bar(plt_x, plt_y)
        for classid, distribution_ratio in zip(plt_x, plt_y): 
            plt.text(classid, distribution_ratio, '{:.2f}%'.format(distribution_ratio*100), ha='center', va='bottom')  # 在(classid, distribution_ratio)显示具体数值
        plt.title('every class distribution')
        plt.xlabel('classid')
        plt.ylabel('distribution ratio')
        plt.savefig(self.img_save_path + class_distribution_name)
        plt.close(0) 

        # 各个类别的数量占比绘制
        plt_x = []
        plt_y = []
        object_num = 0
        for i in self.dict_class_num_distribution:
            object_num += self.dict_class_num_distribution[i]
        for i in range(1, len(self.colors)+1):  #i: 1 ~ classnum
            if i in self.dict_class_num_distribution:
                plt_x.append('%g' % i)
                plt_y.append(self.dict_class_num_distribution[i] / object_num)
        fig = plt.figure(0)
        plt.bar(plt_x, plt_y)
        for classid, distribution_ratio in zip(plt_x, plt_y): 
            plt.text(classid, distribution_ratio, '{:.2f}%'.format(distribution_ratio*100), ha='center', va='bottom')  # 在(classid, distribution_ratio)显示具体数值
        plt.title('objects distribution')
        plt.xlabel('classid')
        plt.ylabel('distribution ratio')
        plt.savefig(self.img_save_path + objects_distribution_name)
        plt.close(0)
        
        if self.img_aug_id < self.save_img_num: 
            filename = self.img_save_path + ('img_augment%g.jpg' % self.img_aug_id)  # filename
            self.img_aug_id += 1
            img = copy.deepcopy(results['img'])       # img(h, w, 3) 未归一化
            polys = results['gt_bboxes']  # results['gt_bboxes'] (n, 8)
            labels = results['gt_labels'] # results['gt_labels'] (n)
            # visulize the oriented boxes
            for i, bbox in enumerate(polys):   
                cls_index = labels[i] - 1
                # box_list.size(4, 2)
                box_list = np.array([(bbox[0], bbox[1]), (bbox[2], bbox[3]), (bbox[4], bbox[5]), (bbox[6], bbox[7])], np.int32)
                cv2.drawContours(image=img, contours=[box_list], contourIdx=-1, color=self.colors[int(cls_index)], thickness=self.thickness)
            cv2.imwrite(filename, img)
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(img_save_path={})'.format(
            self.img_save_path,
            self.save_img_num,
            self.colors)