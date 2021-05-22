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
It is recommended to symlink the dataset root to $orientedreppoints/data. If your folder structure is different, you may need to change the corresponding paths in config files.
```
orientedreppoints
|——mmdet
|——tools
|——configs
|——data
|  |——dota
|  |  |——trainval_split
|  |  |  |——images
|  |  |  |——labelTxt
|  |  |  |——trainval.json
|  |  |——test_split
|  |  |  |——images
|  |  |  |——test.json
|  |——HRSC2016(OPTINAL)
|  |  |——Train
|  |  |  |——images
|  |  |  |——labelTxt
|  |  |  |——train.txt
|  |  |  |——trainval.json
|  |  |——Test
|  |  |  |——images
|  |  |  |——test.txt
|  |  |  |——test.json
|  |——UCASAOD(OPTINAL)
|  |  |——Train
|  |  |  |——images
|  |  |  |——labelTxt
|  |  |  |——train.txt
|  |  |  |——trainval.json
|  |  |——Test
|  |  |  |——images
|  |  |  |——test.txt
|  |  |  |——test.json
```
Note:
* `train.txt` and `test.txt` in HRSC2016 and UCASAOD are `.txt` files recording image names without extension.
* Without the pre-divided `train`，`test`, and `val` sub-dataset, the partition of UCASAOD dataset follows the [rep](https://github.com/ming71/UCAS-AOD-benchmark).




