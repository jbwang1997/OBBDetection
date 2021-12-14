// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once

#include "./cpu/vision.h"

#ifdef WITH_CUDA
#include "./cuda/vision.h"
#endif

// Interface for Python
at::Tensor ARF_forward(const at::Tensor& weight,
                       const at::Tensor& indices) {
  if (weight.type().is_cuda()) {
#ifdef WITH_CUDA
  return ARF_forward_cuda(weight, indices);
#else
  AT_ERROR("Not compiled with GPU support");
#endif
  }
  return ARF_forward_cpu(weight, indices);
}

at::Tensor ARF_backward(const at::Tensor& indices,
                        const at::Tensor& gradOutput) {
  if (gradOutput.type().is_cuda()) {
#ifdef WITH_CUDA
  return ARF_backward_cuda(indices, gradOutput);
#else
  AT_ERROR("Not compiled with GPU support");
#endif
  }
  return ARF_backward_cpu(indices, gradOutput);
  AT_ERROR("Not implemented on the CPU");
}