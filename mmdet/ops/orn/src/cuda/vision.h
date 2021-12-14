// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
// #include <torch/extension.h>
#include <torch/serialize/tensor.h>

typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;


std::tuple<at::Tensor, at::Tensor> RIE_forward_cuda(const at::Tensor& feature,
                                                    const uint8 nOrientation);

at::Tensor RIE_backward_cuda(const at::Tensor& mainDirection,
                             const at::Tensor& gradOutput,
                             const uint8 nOrientation);

at::Tensor ARF_forward_cuda(const at::Tensor& weight,
                            const at::Tensor& indices);

at::Tensor ARF_backward_cuda(const at::Tensor& indices,
                             const at::Tensor& gradOutput);