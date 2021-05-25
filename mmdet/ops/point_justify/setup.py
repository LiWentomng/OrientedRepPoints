from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import subprocess
import numpy as np

## 判断点集与多边形的包含关系
#  输入第一个参数为 points  [m,2]
#  输入第二个参数为 polygon [n,8]
#  输出为 [m,n]  意义为第 m 个 points 与 第 n 个 polygon 的包含关系 1-在内部 0-在外部
setup(
    name='points_justify',   # import 的 modules 名字
    ext_modules=[
        CUDAExtension('points_justify', [
            'points_justify.cpp',
            'points_justify_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})

