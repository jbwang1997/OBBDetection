// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once

#include "./cpu/vision.h"

#ifdef WITH_CUDA
#include "./cuda/vision.h"
#endif

// Interface for Python
std::tuple<at::Tensor, at::Tensor> RIE_forward(const at::Tensor& feature,
                                               const uint8 nOrientation) {
  if (feature.type().is_cuda()) {
#ifdef WITH_CUDA
  return RIE_forward_cuda(feature, nOrientation);
#else
  AT_ERROR("Not compiled with GPU support");
#endif
  }
  return RIE_forward_cpu(feature, nOrientation);
}

at::Tensor RIE_backward(const at::Tensor& mainDirection,
                        const at::Tensor& gradOutput,
                        const uint8 nOrientation) {
  if (gradOutput.type().is_cuda()) {
#ifdef WITH_CUDA
  return RIE_backward_cuda(mainDirection, gradOutput, nOrientation);
#else
  AT_ERROR("Not compiled with GPU support");
#endif
  }
  return RIE_backward_cpu(mainDirection, gradOutput, nOrientation);
  AT_ERROR("Not implemented on the CPU");
}