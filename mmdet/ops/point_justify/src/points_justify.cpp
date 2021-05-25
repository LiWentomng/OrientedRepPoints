#include <torch/extension.h> //包含了数据格式的定义 相当于一个数据接口文件
#include <cmath>
#include <iostream>
#include <vector>

int PointsJFLaucher(const at::Tensor points, const at::Tensor polygons,
                const int rows, const int cols, at::Tensor output);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


int pointsJf(at::Tensor points, at::Tensor polygons, at::Tensor output) {
    CHECK_INPUT(points);
    CHECK_INPUT(polygons);
    CHECK_INPUT(output);

    //检查points格式 [ ,2]
    int size_points = points.size(1);
    if(size_points != 2){
        printf("wrong points size\n");
        return 0;
    }
    //检查polygons格式 [ ,8]
    int size_polygons = points.size(1);
    if(size_polygons != 2){
        printf("wrong polygons size\n");
        return 0;
    }

    int rows = points.size(0);
    int cols = polygons.size(0);
    PointsJFLaucher(points, polygons, rows, cols, output);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pointsJf", &pointsJf, "Justify a point is in polygon or not");  //该对象对应的方法名   函数指针  说明  
}