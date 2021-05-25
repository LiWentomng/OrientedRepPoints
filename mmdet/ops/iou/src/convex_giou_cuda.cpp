// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/extension.h>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")

at::Tensor convex_giou_cuda(const at::Tensor pred, const at::Tensor target);

at::Tensor convex_giou(const at::Tensor pred, const at::Tensor target) {
  CHECK_CUDA(pred);
  CHECK_CUDA(target);
  
  if (pred.numel() == 0 || target.numel() == 0)
    return at::empty({0}, pred.options().dtype(at::kFloat).device(at::kCPU));
    
  return convex_giou_cuda(pred, target);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("convex_giou", &convex_giou, "convex generally iou");
}
