# Oriented RepPoints for Aerial Object Detection
> Wentong Li, Yijie Chen, Kaixuan Hu, Jianke Zhu* ([Arxiv](https://arxiv.org/pdf/2105.11111v4.pdf))

<img src="https://github.com/LiWentomng/OrientedRepPoints/blob/main/docs/overallnetwork.png" width="800px">

# 
* Based on OrientedRepPoints detector, the **2nd**  and **3rd** Places are achieved on the Task 2 and Task 1 respectively in the *“2021 challenge of Learning to Understand Aerial Images([LUAI](https://captain-whu.github.io/LUAI2021/tasks.html))”*. **The detailed codes and introductions about it, please refer to this [repository](https://github.com/hukaixuan19970627/OrientedRepPoints_DOTA) and [知乎](https://zhuanlan.zhihu.com/p/422764914)**.

## Update
* **About the detailed installation, please see this [CSDN Blog](https://blog.csdn.net/SSSlasH/article/details/125255955).** (Thanks for author@SSSlasH of this blog).

* The code for [MMRotate](https://github.com/open-mmlab/mmrotate) is available now.
 
* [RepPoints](https://github.com/microsoft/RepPoints) + our **APAA** can obtain **+2.5AP** (36.3 to 38.8) improvement with R-50 on **COCO** dataset for general object detection.


# Installation
Please refer to ![install.md](https://github.com/LiWentomng/OrientedRepPoints/blob/main/docs/install.md) for installation and dataset preparation.


# Getting Started 
This repo is based on ![mmdetection](https://github.com/open-mmlab/mmdetection). Please see ![getting_started.md](https://github.com/LiWentomng/OrientedRepPoints/blob/main/docs/getting_started.md) for the basic usage.

# Results and Models
The results on DOTA test set are shown in the table below. More detailed results please see the paper.

  Model| Backbone  |data aug(HSV+Rotation)| mAP | model| log
 ----  | ----- | ------ |------| ------ | ------  
 OrientedReppoints| R-50|  |75.97 |[model](https://drive.google.com/file/d/13c56u9IFRRdHH-YNmQfqb1y11f7xPfCR/view?usp=sharing) | [log](https://drive.google.com/file/d/1_lrj3gV27iM0v95AnSCRHUZDZWkdJFS_/view?usp=sharing)
 OrientedReppoints| R-101| |76.52 |[model](https://drive.google.com/file/d/1otXS3w0LVopsBKxyYbyQhF6mFDtTIJFX/view?usp=sharing) | [log](https://drive.google.com/file/d/1MgJ7A9INaP3iocy1MQSS1SA6gyIvnTJX/view?usp=sharing)
 OrientedReppoints| Swin-Tiny|  √  | 78.11|[model](https://drive.google.com/file/d/1B03dBSXU9GFGRM8XiyQo2aw6yGnCgB8r/view?usp=sharing) | [log](https://drive.google.com/file/d/1lt5UkBPVM7am6asySRWohXSRK_tGwxV8/view?usp=sharing)

Note: 
* The pretrained model--*swin_tiny_patch4_window7_224* of [Swin-Tiny](https://github.com/microsoft/Swin-Transformer) for pytorch1.4.0 is [here](https://drive.google.com/file/d/1ad4lxks68vngs_pCaqs9w_L-fGvtR7nQ/view?usp=sharing).
* We recommend to use our demo configs with 4 GPUs.
* The results are performed on the original DOTA images with 1024x1024 patches. 
* The scale jitter is employed during training. More details see the paper.


The mAOE results on DOTA val set are shown in the table below.

  Model| Backbone | mAOE | Download
 ----  | ----- | ------  | ------
 OrientedReppoints| R-50| 5.93° |[model](https://drive.google.com/file/d/1lGHehF57ObkAt0i9FITkp5yS6ULBZQjx/view?usp=sharing)

 Note：Orientation error evaluation (mAOE) is calculated on the val subset(only train subset for training).

# Visual results
The visualization code for oriented bounding boxes and learning points is ![here](https://github.com/LiWentomng/OrientedRepPoints/blob/main/tools/parse_pkl/show_learning_points_and_boxes.py).

* Oriented bounding box

<img src="https://github.com/LiWentomng/OrientedRepPoints/blob/main/docs/visualization.png" width="800px">


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

[RepPoints](https://github.com/microsoft/RepPoints)

[AerialDetection](https://github.com/dingjiansw101/AerialDetection)

[BeyondBoundingBox](https://github.com/sdl-guozonghao/beyondboundingbox)




