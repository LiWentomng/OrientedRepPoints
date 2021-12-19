# [Oriented RepPoints for Aerial Object Detection](https://arxiv.org/pdf/2105.11111.pdf)
> Wentong Li, Yijie Chen, Kaixuan Hu, Jianke Zhu*

![](https://github.com/LiWentomng/OrientedRepPoints/blob/main/docs/visualization.png)

## *News*
* The codes have been refined, and [Swin-transformer](https://github.com/microsoft/Swin-Transformer) backbone has been included in this repo.
* Based on OrientedRepPoints detector, the **2rd**  and **3rd** Place are achieved on the Task 2 and Task 1 respectively in the *“2021 challenge of Learning to Understand Aerial Images([LUAI](https://captain-whu.github.io/LUAI2021/tasks.html))”*. The detailed codes and introduction about it, please refer to this [repository](https://github.com/hukaixuan19970627/OrientedRepPoints_DOTA).

### *Learning Points of Oriented RepPoints*
![Learning Adative Points](https://github.com/LiWentomng/OrientedRepPoints/blob/main/docs/learning_points.png)

# Installation
## Requirements
* Linux
* Python 3.7+ 
* PyTorch1.3 or higher
* CUDA 9.0 or higher
* mmdet==1.1.0
* mmcv-full==1.3.15
* GCC 4.9 or higher
* NCCL 2

We have tested the following versions of OS and softwares：
* OS：Ubuntu 16.04
* CUDA: 10.0
* Python 3.7
* PyTorch 1.3.1

## Install 
a. Create a conda virtual environment and activate it.  
```
conda create -n orientedreppoints python=3.7 -y 
source activate orientedreppoints
```
b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/previous-versions/), e.g.,
```
conda install pytorch=1.3 torchvision cudatoolkit=10.0 -c pytorch
```
c. Install orientedreppoints.

```python 
cd OrientedRepPoints
pip install -r requirements.txt
python setup.py develop  #or "pip install -v -e ."
```

## Install DOTA_devkit

```
sudo apt-get install swig
```
```
cd DOTA_devkit
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```
## Prepare dataset
It is recommended to symlink the dataset root to $orientedreppoints/data. If your folder structure is different, you may need to change the corresponding paths in config files.
```
orientedreppoints
|——mmdet
|——tools
|——configs
|  |--dota
|  |  |--orientedreppoints_r50.py
|  |  |--orientedreppoints_r101.py
|  |  |--orientedreppoints_swin-t.py
|  |--hrsc2016
|  |--ucas-aod
|  |--dior-r
|——data
|  |——dota
|  |  |——trainval_split
|  |  |  |——images
|  |  |  |——labelTxt
|  |  |  |——trainval.json
|  |  |——test_split
|  |  |  |——images
|  |  |  |——test.json
|  |——HRSC2016
|  |  |——Train
|  |  |  |——images
|  |  |  |——labelTxt
|  |  |  |——trainval.txt
|  |  |  |——trainval.json
|  |  |——Test
|  |  |  |——images
|  |  |  |——test.txt
|  |  |  |——test.json
|  |——UCAS-AOD
|  |——DIOR-R
```
Note:
* `trainval.txt` and `test.txt` in HRSC2016, UCASAOD and DIOR-R are `.txt` files recording image names without extension.


# Getting Started 
Our code is based on ![mmdetection](https://github.com/open-mmlab/mmdetection). 

## Train a model

1. Train  with a single GPU 

```shell
CUDA_VISIBLE_DEVICES=0 python tools/train.py  configs/dota/orientedreppoints_r50.py
```

2. Train with multiple GPUs

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/dota/orientedreppoints_r50.py 4
```
All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dirs` in the config file.

## Inferenece 
We provide the testing scripts to evaluate the trained model.

Examples:

1. Test OrientedRepPoints with single GPU.

```shell
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/dota/orientedreppoints_r50.py \
    work_dirs/orientedreppoints_r50/epoch_40.pth \ 
    --out work_dirs/orientedreppoints_r50/results.pkl

```
2. Test OrientedRepPoints with 4 GPUs.
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_test.sh configs/dota/orientedrepoints_r50.py \
    work_dirs/orientedreppoints_r50/epoch_40.pth 4 \ 
    --out work_dirs/orientedreppoints_r50/results.pkl
```

*If you want to evaluate the mAP results on DOTA test-dev, please parse the results.pkl, and merge the txt results. and zip the files  and submit it to the  [evaluation server](https://captain-whu.github.io/DOTA/index.html).



## Acknowledgement

Here are some great resources we benefit:

[AerialDetection](https://github.com/dingjiansw101/AerialDetection)

[BeyondBoundingBox](https://github.com/SDL-GuoZonghao/BeyondBoundingBox)


# Citation 

```shell
@article{li2021oriented,
	title="Oriented RepPoints for Aerial Object Detection.",
	author="Wentong {Li}, Yijie {Chen}, Kaixuan {Hu}, Jianke {Zhu}.",
	journal="arXiv preprint arXiv:2105.11111",
	year="2021"
}

```


