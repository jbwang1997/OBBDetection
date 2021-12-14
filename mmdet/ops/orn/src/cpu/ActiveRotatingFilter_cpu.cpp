#include "./vision.h"

template <typename T>
void ARF_forward_cpu_kernel(
  const T* weightData,
  const uint8* indicesData,
  const uint16 nOutputPlane,
  const uint16 nInputPlane,
  const uint8 nOrientation,
  const uint8 kH,
  const uint8 kW,
  const uint8 nRotation,
  T* outputData)
{
  const uint16 nEntry = nOrientation * kH * kW;
  uint16 i, j, l;
  uint8 k;

#pragma omp parallel for private(i, j, l, k)
  for (i = 0; i < nOutputPlane; i++) {
    for (j = 0; j < nInputPlane; j++) {
      for (l = 0; l < nEntry; l++) {
        uint16 weightIndex = i * nInputPlane * nEntry
                             + j * nEntry
                             + l;
        T val = *(weightData + weightIndex);
        // T val = *(weightData++);
        for (k = 0; k < nRotation; k++) {
          uint16 index = (uint16)(*(indicesData + l * nRotation + k)) - 1;
          T *target = outputData + i * (nRotation * nInputPlane * nEntry)
                                 + k * (nInputPlane * nEntry)
                                 + j * (nEntry)
                                 + index;
          *target = val;
        }
      }
    }
  }
}

template <typename T>
void ARF_backward_cpu_kernel(
  const uint8* indicesData,
  const T* gradOutputData,
  const uint16 nOutputPlane,
  const uint16 nInputPlane,
  const uint8 nOrientation,
  const uint8 kH,
  const uint8 kW,
  const uint8 nRotation,
  T* gradInputData)
{
  const uint16 nEntry = nOrientation * kH * kW;
  uint16 i, j, l;
  uint8 k;

#pragma omp parallel for private(i, j, l, k)
  for (i = 0; i < nOutputPlane; i++) {
    for (j = 0; j < nInputPlane; j++) {
      for (l = 0; l < nEntry; l++) {
        uint16 gradInputIndex = i * nInputPlane * nEntry
                                + j * nEntry
                                + l;
        T *val = gradInputData + gradInputIndex;
        // T *val = gradInputData++;
        *val = 0;
        for (k = 0; k < nRotation; k++) {
          uint16 index = (uint16)(*(indicesData + l * nRotation + k)) - 1;
          const T *target = gradOutputData + i * (nRotation * nInputPlane * nEntry)
                                           + k * (nInputPlane * nEntry)
                                           + j * (nEntry)
                                           + index;
          *val = *val + *target;
        }
      }
    }
  }
}


at::Tensor ARF_forward_cpu(const at::Tensor& weight,
                           const at::Tensor& indices) {
  AT_ASSERTM(weight.ndimension() == 5, "only supports a batch of ARFs.");
  AT_ASSERTM(!weight.type().is_cuda(), "input must be a CPU tensor");
  AT_ASSERTM(!indices.type().is_cuda(), "rois must be a CPU tensor");

  const uint16 nOutputPlane = weight.size(0);
  const uint16 nInputPlane = weight.size(1);
  const uint8 nOrientation = weight.size(2);
  const uint8 kH = weight.size(3);
  const uint8 kW = weight.size(4);
  const uint8 nRotation = indices.size(3);

  auto output = at::empty({nOutputPlane * nRotation, nInputPlane * nOrientation, kH, kW}, weight.options());
  
  AT_DISPATCH_FLOATING_TYPES(weight.type(), "ARF_forward", [&] {
    ARF_forward_cpu_kernel<scalar_t>(
         weight.data<scalar_t>(),
         indices.data<uint8_t>(),
         nOutputPlane,
         nInputPlane,
         nOrientation,
         kH,
         kW,
         nRotation,
         output.data<scalar_t>());
  });
  return output;
}


at::Tensor ARF_backward_cpu(const at::Tensor& indices,
                            const at::Tensor& gradOutput) {
  AT_ASSERTM(!indices.type().is_cuda(), "input must be a CPU tensor");
  AT_ASSERTM(!gradOutput.type().is_cuda(), "rois must be a CPU tensor");

  const uint8 nOrientation = indices.size(0);
  const uint8 kH = indices.size(1);
  const uint8 kW = indices.size(2);
  const uint8 nRotation = indices.size(3);
  const uint16 nOutputPlane = gradOutput.size(0) / nRotation;
  const uint16 nInputPlane = gradOutput.size(1) / nOrientation;

  at::Tensor gradInput = at::zeros({nOutputPlane, nInputPlane, nOrientation, kH, kW}, gradOutput.options());

  // handle possibly empty gradients
  if (gradOutput.numel() == 0) {
    return gradInput;
  }

  AT_DISPATCH_FLOATING_TYPES(gradOutput.type(), "ARF_backward", [&] {
    ARF_backward_cpu_kernel<scalar_t>(
         indices.data<uint8_t>(),
         gradOutput.data<scalar_t>(),
         nOutputPlane,
         nInputPlane,
         nOrientation,
         kH,
         kW,
         nRotation,
         gradInput.data<scalar_t>());
  });
  return gradInput;
}