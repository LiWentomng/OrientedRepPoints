// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#pragma once
#include <torch/extension.h>
#include <torch/types.h>


at::Tensor box_iou_rotated_cpu(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2);

#ifdef WITH_CUDA
at::Tensor box_iou_rotated_cuda(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2);
#endif

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
inline at::Tensor box_iou_rotated(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2) {
  assert(boxes1.device().is_cuda() == boxes2.device().is_cuda());
  if (boxes1.device().is_cuda()) {
#ifdef WITH_CUDA
    return box_iou_rotated_cuda(boxes1, boxes2);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  return box_iou_rotated_cpu(boxes1, boxes2);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("box_iou_rotated", &box_iou_rotated, "IoU for rotated boxes");
}