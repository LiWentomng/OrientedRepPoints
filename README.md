# [Oriented RepPoints for Aerial Object Detection](https://arxiv.org/pdf/2105.11111.pdf)
> Wentong Li, Yijie Chen, Kaixuan Hu, Jianke Zhu*


# 
* Based on OrientedRepPoints detector, the **2rd**  and **3rd** Places are achieved on the Task 2 and Task 1 respectively in the *“2021 challenge of Learning to Understand Aerial Images([LUAI](https://captain-whu.github.io/LUAI2021/tasks.html))”*. The detailed codes and introductions about it, please refer to this [repository](https://github.com/hukaixuan19970627/OrientedRepPoints_DOTA).

# Installation
Please refer to ![install.md](https://github.com/LiWentomng/OrientedRepPoints/blob/main/docs/install.md) for installation and dataset preparation.


# Getting Started 
This repo is based on ![mmdetection](https://github.com/open-mmlab/mmdetection). Please see ![getting_started.md](https://github.com/LiWentomng/OrientedRepPoints/blob/main/docs/getting_started.md) for the basic usage.

# Results and Models
The results on DOTA test set are shown in the table below. More detailed results please see the paper.

  Model| Backbone  |aug test| mAP | model| log
 ----  | ----- | ------  | ------| ------ | ------  
 OrientedReppoints| R-50| |75.97 |[model](https://drive.google.com/file/d/13c56u9IFRRdHH-YNmQfqb1y11f7xPfCR/view?usp=sharing) | [log](https://drive.google.com/file/d/1_lrj3gV27iM0v95AnSCRHUZDZWkdJFS_/view?usp=sharing)
 OrientedReppoints| R-101| |76.52 |[model](https://drive.google.com/file/d/1otXS3w0LVopsBKxyYbyQhF6mFDtTIJFX/view?usp=sharing) | [log]()
 OrientedReppoints| Swin-Tiny|    | 77.30|[model](https://drive.google.com/file/d/1dXDu1xrGg2OmISOXGiJlngNKtiELMCqT/view?usp=sharing) | [log](https://drive.google.com/file/d/1XaDbaV0zbi3lwmWqTQKpelfDEmsvLv6v/view?usp=sharing)
 OrientedReppoints| Swin-Tiny| √  | 77.63|above  |above

The mAOE results on DOTA val set are shown in the table below(password:aabb).

  Model| Backbone | mAOE | Download
 ----  | ----- | ------  | ------
 OrientedReppoints| R-50| 5.93° |[model](https://drive.google.com/file/d/1lGHehF57ObkAt0i9FITkp5yS6ULBZQjx/view?usp=sharing)

 Note：
 * Wtihout the ground-truth of test dataset, the mAOE (mean average orientation error) of orientation evaluation is calculated on the val subset(original train subset for training).

 
# Visual results
The visual results of learning points and the oriented bounding boxes. The visualization code is ![show_learning_points_and_boxes.py](https://github.com/LiWentomng/OrientedRepPoints/blob/main/tools/parse_pkl/show_learning_points_and_boxes.py).

* Learning points

![Learning Adative Points](https://github.com/LiWentomng/OrientedRepPoints/blob/main/docs/learning_points.png)

* Oriented bounding box



# Citation
```shell
@inproceeding{orientedreppoints,
	title="Oriented RepPoints for Aerial Object Detection.",
	author="Wentong {Li}, Yijie {Chen}, Kaixuan {Hu}, Jianke {Zhu}.",
	journal="The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
	year="2022"
}
```


#  Acknowledgements
Here are some great resources we benefit. We would espeicially thank the authors of:

[MMdetection](https://github.com/open-mmlab/mmdetection)

[AerialDetection](https://github.com/dingjiansw101/AerialDetection)

[BeyoundBoundingBox](https://github.com/sdl-guozonghao/beyondboundingbox)




