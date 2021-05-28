# Oriented RepPoints for Aerial Object Detection
![图片](https://user-images.githubusercontent.com/32033843/119212550-b44da380-baeb-11eb-9de2-61ce0d812131.png)

The code for the implementation of “Oriented RepPoints”, [arXiv preprint](https://arxiv.org/abs/2105.11111).

This repo is based on ![mmdetection](https://github.com/open-mmlab/mmdetection).

# Introduction
Oriented RepPoints employs a set of adaptive points to capture the geometric and spatial information of the arbitrary-oriented objects, which is able to automatically arrange themselves over the object in a spatial and semantic scenario. To facilitate the supervised learning, the oriented conversion function is proposed to explicitly map the adaptive point set into an oriented bounding box. Moreover, we introduce an effective quality assessment measure to select the point set samples for training, which can choose the representative items with respect to their potentials on orientated object detection. Furthermore, we suggest a spatial constraint to penalize the outlier points outside the groundtruth bounding box. In addition to the traditional evaluation metric mAP focusing on overlap ratio, we propose a new metric mAOE to measure the orientation accuracy that is usually neglected in the previous studies on oriented object detection. Experiments on three widely used datasets including DOTA, HRSC2016 and UCAS-AOD demonstrate that our proposed approach is effective. 


# Installation
Please refer to ![install.md](https://github.com/LiWentomng/OrientedRepPoints/blob/main/docs/install.md) for installation and dataset preparation.


# Getting Started 
Please see ![getting_started.md](https://github.com/LiWentomng/OrientedRepPoints/blob/main/docs/getting_started.md) for the basic usage of MMDetection.

# Results and Models
The results on DOTA test set are shown in the table below(password:aabb). More detailed results please see the paper.

  Model| Backbone  | MS | Rotate | mAP | Download
 ----  | ----- | ------  | ------| ------ | ------  
 OrientedReppoints| R-50| - | -| 75.68 |[model](https://pan.baidu.com/s/1fCgmpd3MWoCbI80wYwtV2w)
 OrientedReppoints| R-101| - | √| 76.21 |[model](https://pan.baidu.com/s/1WN2QKMR6vrTzrJGCcukt8A)
 OrientedReppoints| R-101| √ | √ | 78.12|[model](https://pan.baidu.com/s/1Rv2ujQEt56R9nw-QjJlMIg)
 

The mAOE results on DOTA val set are shown in the table below(password:aabb).
  Model| Backbone | mAOE | Download
 ----  | ----- | ------  | ------
 OrientedReppoints| R-50| 5.93° |[model](https://pan.baidu.com/s/1TeHDeuVTKpXd5KdYY51TUA)


 Note：
 * Wtihout the ground-truth of test subset, the mAOE of orientation evaluation is calculated on the val subset(original train subset for training).
 * The orientation (angle) of an aerial object is define as below, the detail of mAOE, please see the paper. The code of mAOE is [mAOE_evaluation.py](https://github.com/LiWentomng/OrientedRepPoints/blob/main/DOTA_devkit/mAOE_evaluation.py).
 ![微信截图_20210522135042](https://user-images.githubusercontent.com/32033843/119216186-be2fd080-bb04-11eb-9736-1f82c6666171.png)

 
# Visual results
The visual results of learning points and the oriented bounding box.
* Learning points

![Learning Points](https://user-images.githubusercontent.com/32033843/119213326-e44b7580-baf0-11eb-93a6-c86fcf80be58.png)

* Oriented bounding box

![Oriented Box](https://user-images.githubusercontent.com/32033843/119213335-edd4dd80-baf0-11eb-86db-459fe2a14735.png)


# Citation
```shell
@article{Li2021oriented,
  title={Oriented RepPoints for Aerial Object Detection},
  author={Wentong Li and Jianke Zhu},
  journal={arXiv preprint arXiv:2105.11111},
  year={2021}
}
```


#  Acknowledgements
We have used utility functions from other wonderful open-source projects, we would espeicially thank the authors of:

[MMdetection](https://github.com/open-mmlab/mmdetection)

[DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)

[AerialDetection](https://github.com/dingjiansw101/AerialDetection)

[BeyoundBoundingBox](https://github.com/sdl-guozonghao/beyondboundingbox)




