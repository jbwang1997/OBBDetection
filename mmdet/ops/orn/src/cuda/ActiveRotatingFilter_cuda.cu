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
__global__ void ARF_forward_cuda_kernel(
  const long nthreads, 
  const Dtype* weight_data,
  const uint8* indices_data,
  const uint16 nInputPlane,
  const uint16 nOutputPlane,
  const uint8 nOrientation,
  const uint8 nRotation,
  const uint16 nEntry,
  Dtype* output_data) 
{
  CUDA_KERNEL_LOOP(n, nthreads) {
    uint16 l = n % nEntry;
    uint16 j = (n / nEntry) % nInputPlane;
    uint16 i = n / nEntry / nInputPlane;
    uint8 k;
    Dtype val = *(weight_data + n);
    for (k = 0; k < nRotation; k++) {
      uint16 index = (uint16)(*(indices_data + l * nRotation + k)) - 1;
      Dtype *target = output_data + i * (nRotation * nInputPlane * nEntry)
                                  + k * (nInputPlane * nEntry)
                                  + j * (nEntry)
                                  + index;
      *target = val;
    }
  }
}

template <typename Dtype>
__global__ void ARF_backward_cuda_kernel(
  const long nthreads, 
  const Dtype* gradWeight_data,
  const uint8* indices_data,
  const uint16 nInputPlane,
  const uint16 nOutputPlane,
  const uint8 nOrientation,
  const uint8 nRotation,
  const uint16 nEntry,
  Dtype* weight_data) 
{
  CUDA_KERNEL_LOOP(n, nthreads) {
      uint16 l = n % nEntry;
      uint16 j = (n / nEntry) % nInputPlane;
      uint16 i = n / nEntry / nInputPlane;
      uint8 k;
      Dtype *val = weight_data + n;
      *val = 0;
      for (k = 0; k < nRotation; k++) {
          uint16 index = (uint16)(*(indices_data + l * nRotation + k)) - 1;
          Dtype target = *(gradWeight_data + i * (nRotation * nInputPlane * nEntry)
                                           + k * (nInputPlane * nEntry)
                                           + j * (nEntry)
                                           + index);
          *val = *val + target;
      }
  }
}


at::Tensor ARF_forward_cuda(const at::Tensor& weight,
                            const at::Tensor& indices) {
  AT_ASSERTM(weight.ndimension() == 5, "only supports a batch of ARFs.");
  AT_ASSERTM(weight.type().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(indices.type().is_cuda(), "rois must be a CUDA tensor");

  const uint16 nOutputPlane = weight.size(0);
  const uint16 nInputPlane = weight.size(1);
  const uint8 nOrientation = weight.size(2);
  const uint8 kH = weight.size(3);
  const uint8 kW = weight.size(4);
  const uint8 nRotation = indices.size(3);

  auto output = at::empty({nOutputPlane * nRotation, nInputPlane * nOrientation, kH, kW}, weight.options());
  const uint16 nEntry = nOrientation * kH * kW;
  const long output_size = nOutputPlane * nInputPlane * nEntry;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(std::min(THCCeilDiv(output_size, 512L), 4096L));
  dim3 block(512);

  if (output.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(weight.type(), "ARF_forward", [&] {
    ARF_forward_cuda_kernel<scalar_t><<<grid, block, 0, stream>>>(
         output_size,
         weight.contiguous().data<scalar_t>(),
         indices.contiguous().data<uint8_t>(),
         nInputPlane,
         nOutputPlane,
         nOrientation,
         nRotation,
         nEntry,
         output.contiguous().data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return output;
}


at::Tensor ARF_backward_cuda(const at::Tensor& indices,
                             const at::Tensor& gradOutput) {
  AT_ASSERTM(indices.type().is_cuda(), "input must be a CPU tensor");
  AT_ASSERTM(gradOutput.type().is_cuda(), "rois must be a CPU tensor");

  const uint8 nOrientation = indices.size(0);
  const uint8 kH = indices.size(1);
  const uint8 kW = indices.size(2);
  const uint8 nRotation = indices.size(3);
  const uint16 nOutputPlane = gradOutput.size(0) / nRotation;
  const uint16 nInputPlane = gradOutput.size(1) / nOrientation;

  auto gradWeight = at::zeros({nOutputPlane, nInputPlane, nOrientation, kH, kW}, gradOutput.options());
  const uint16 nEntry = nOrientation * kH * kW;
  const long count = nOutputPlane * nInputPlane * nEntry;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv(count, 512L), 4096L));
  dim3 block(512);

  // handle possibly empty gradients
  if (gradOutput.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return gradWeight;
  }

  AT_DISPATCH_FLOATING_TYPES(gradOutput.type(), "ARF_backward", [&] {
    ARF_backward_cuda_kernel<scalar_t><<<grid, block, 0, stream>>>(
         count,
         gradOutput.contiguous().data<scalar_t>(),
         indices.contiguous().data<uint8_t>(),
         nInputPlane,
         nOutputPlane,
         nOrientation,
         nRotation,
         nEntry,
         gradWeight.contiguous().data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return gradWeight;
}