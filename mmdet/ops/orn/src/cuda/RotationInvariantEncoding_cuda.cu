// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include "./vision.h"

// #define FLT_MAX 3.402823466e+38F

// TODO make it in a common file
#define CUDA_KERNEL_LOOP(i, n)                               \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename Dtype>
__global__ void RIE_forward_cuda_kernel(
  const uint32 nthreads, 
  const Dtype* feature_data,
  const uint16 nBatch,
  const uint16 nFeature,
  const uint8 nOrientation,
  uint8* mainDirection_data,
  Dtype* aligned_data) 
{
  CUDA_KERNEL_LOOP(n, nthreads) {
    const uint16 j = n % nFeature;
    const uint16 i = n / nFeature;
    uint8 l;
    
    uint8 *direction = mainDirection_data + i * nFeature + j;
    Dtype maxVal = -FLT_MAX;
    for (l = 0; l < nOrientation; l++) {
      Dtype val = *(feature_data + i * (nFeature * nOrientation)
                                 + j * (nOrientation)
                                 + l);
      if (val > maxVal) {
        maxVal = val;
        *direction = l;
      }
    }
    for (l = 0; l < nOrientation; l++) {
      Dtype src = *(feature_data + i * (nFeature * nOrientation)
                                 + j * (nOrientation)
                                 + l);
      uint8 alignedIndex = ((l - (uint8)*direction) + nOrientation) % nOrientation;
      Dtype *target = aligned_data + i * (nFeature * nOrientation)
                                   + j * (nOrientation)
                                   + alignedIndex;
      *target = src;
    } 
  }
}

template <typename Dtype>
__global__ void RIE_backward_cuda_kernel(
  const uint32 nthreads, 
  const Dtype* aligned_data,
  const uint8* mainDirection_data,
  const uint16 nBatch,
  const uint16 nFeature,
  const uint8 nOrientation,
  Dtype* feature_data) 
{
  CUDA_KERNEL_LOOP(n, nthreads) {
    uint8 l;
    const uint16 j = n % nFeature; 
    const uint16 i = n / nFeature;
    const uint8 direction = *(mainDirection_data + i * nFeature + j);
    for (l = 0; l < nOrientation; l++) {
      Dtype src = *(aligned_data + i * (nFeature * nOrientation)
                                 + j * (nOrientation)
                                 + l);
      uint8 alignedIndex = (l + direction) % nOrientation;
      Dtype *target = feature_data + i * (nFeature * nOrientation)
                                   + j * (nOrientation)
                                   + alignedIndex;
      *target = src;
    }
  }
}


std::tuple<at::Tensor, at::Tensor> RIE_forward_cuda(const at::Tensor& feature,
                                                    const uint8 nOrientation) {
  AT_ASSERTM(feature.ndimension() == 4, "only supports batch mode.");
  // #MODIFIED
  // AT_ASSERTM(feature.size(2) == 1 && feature.size(3) == 1, "mH x mW should be 1x1.");
  AT_ASSERTM(feature.type().is_cuda(), "input must be a CUDA tensor");

  const uint16 nBatch = feature.size(0);
  const uint16 nChannel = feature.size(1);
  const uint16 nFeature = nChannel / nOrientation;

  auto mainDirection = at::empty({nBatch, nFeature}, feature.options().dtype(at::kByte));
  auto aligned = at::zeros_like(feature);

  const long count = nBatch * nFeature;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(std::min(THCCeilDiv(count, 512L), 4096L));
  dim3 block(512);

  if (mainDirection.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return std::make_tuple(mainDirection, aligned);
  }

  AT_DISPATCH_FLOATING_TYPES(feature.type(), "RIE_forward", [&] {
    RIE_forward_cuda_kernel<scalar_t><<<grid, block, 0, stream>>>(
         count,
         feature.contiguous().data<scalar_t>(),
         nBatch,
         nFeature,
         nOrientation,
         mainDirection.contiguous().data<uint8_t>(),
         aligned.contiguous().data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(mainDirection, aligned);
}


at::Tensor RIE_backward_cuda(const at::Tensor& mainDirection,
                             const at::Tensor& gradOutput,
                             const uint8 nOrientation) {
  AT_ASSERTM(mainDirection.type().is_cuda(), "input must be a CPU tensor");
  AT_ASSERTM(gradOutput.type().is_cuda(), "rois must be a CPU tensor");

  const uint16 nBatch = mainDirection.size(0);
  const uint16 nFeature = mainDirection.size(1);

  auto gradInput = at::zeros_like(gradOutput);

  const long count = nBatch * nFeature;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(std::min(THCCeilDiv(count, 512L), 4096L));
  dim3 block(512);

  // handle possibly empty gradients
  if (gradOutput.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return gradInput;
  }

  AT_DISPATCH_FLOATING_TYPES(gradOutput.type(), "RIE_backward", [&] {
    RIE_backward_cuda_kernel<scalar_t><<<grid, block, 0, stream>>>(
         count,
         gradOutput.contiguous().data<scalar_t>(),
         mainDirection.contiguous().data<uint8_t>(),
         nBatch,
         nFeature,
         nOrientation,
         gradInput.contiguous().data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return gradInput;
}