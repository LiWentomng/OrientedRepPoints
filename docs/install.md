# INSTAllation 
## Requirements
* Linux
* Python 3.7+ 
* PyTorch or higher
* CUDA 9.0 or higher
* mmdet==1.1.0
* ![mmcv](https://github.com/open-mmlab/mmcv)==0.3.1
* GCC 4.9 or higher
* NCCL 2

We have tested the following versions of OS and softwares：
* OS：Ubuntu 16.04
* CUDA: 10.0
* NCCL: 2.1.15/2.2.13/2.3.7/2.4.2
* GCC(G++): 4.9/5.3/5.4/7.3

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
c. Clone the orientedreppoints repository.
```
git clone https://github.com/LiWentomng/OrientedRepPoints.git
cd OrientedRepPoints
```
d. Install orientedreppoints.

```python 
pip install -r requirements.txt
python setup.py develop  #or "pip install -v -e ."
```

## Install DOTA_devkit

```
sudo apt-get install swig
cd DOTA_devkit/polyiou
swig -c++ -python csrc/polyiou.i
python setup.py build_ext --inplace
```

## Prepare dataset






