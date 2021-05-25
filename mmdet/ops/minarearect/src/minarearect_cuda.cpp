// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/extension.h>
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
at::Tensor minareabbox_cuda(const at::Tensor pred);
at::Tensor minareabbox(const at::Tensor pred) {
  CHECK_CUDA(pred);
  if (pred.numel() == 0)
    return at::empty({0}, pred.options().dtype(at::kFloat).device(at::kCPU));
  return minareabbox_cuda(pred);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("minareabbox", &minareabbox, "find minarearect");
}
