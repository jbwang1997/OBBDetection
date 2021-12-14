#include "./vision.h"

#define FLT_MAX 3.402823466e+38F

template <typename T>
void RIE_forward_cpu_kernel(
  const T* feature_data,
  uint8* mainDirection_data,
  T* aligned_data,
  const uint8 nOrientation,
  const uint16 nBatch,
  const uint16 nFeature)
{
  uint16 i;
  uint16 j;
  uint8 l;
  
  #pragma omp parallel for private(i, j, l)
  for (i = 0; i < nBatch; i++) {
    for (j = 0; j < nFeature; j++) {
      uint8 *direction = mainDirection_data + i * nFeature + j;
      T maxVal = -FLT_MAX;
      for (l = 0; l < nOrientation; l++) {
        T val = *(feature_data + i * (nFeature * nOrientation)
                               + j * (nOrientation)
                               + l);
        if (val > maxVal) {
          maxVal = val;
          *direction = l;
        }
      }
      for (l = 0; l < nOrientation; l++) {
        T src = *(feature_data + i * (nFeature * nOrientation)
                               + j * (nOrientation)
                               + l);
        uint8 alignedIndex = (l - (uint8)*direction + nOrientation) % nOrientation;
        T *target = aligned_data + i * (nFeature * nOrientation)
                                 + j * (nOrientation)
                                 + alignedIndex;
        *target = src;
      }
    }
  }
}

template <typename T>
void RIE_backward_cpu_kernel(
  const uint8* mainDirection_data,
  const T* gradOutput_data,
  const uint8 nOrientation,
  const uint16 nBatch,
  const uint16 nFeature,
  T* gradInput_data)
{
  uint16 i;
  uint16 j;
  uint8 l;

  #pragma omp parallel for private(i, j, l)
  for (i = 0; i < nBatch; i++) {
    for (j = 0; j < nFeature; j++) {
      uint8 direction = *(mainDirection_data + i * nFeature + j);
      for (l = 0; l < nOrientation; l++) {
        T src = *(gradOutput_data + i * (nFeature * nOrientation)
                                  + j * (nOrientation)
                                  + l);
        uint8 alignedIndex = (l + direction) % nOrientation;
        T *target = gradInput_data + i * (nFeature * nOrientation)
                                   + j * (nOrientation)
                                   + alignedIndex;
        *target = src;
      }
    }
  }
}


std::tuple<at::Tensor, at::Tensor> RIE_forward_cpu(const at::Tensor& feature,
                                                   const uint8 nOrientation) {
  AT_ASSERTM(feature.ndimension() == 4, "only supports a batch of RIEs.");
  AT_ASSERTM(feature.size(2) == 1 && feature.size(3) == 1, "mH x mW should be 1x1.");
  AT_ASSERTM(!feature.type().is_cuda(), "input must be a CPU tensor");

  const uint16 nBatch = feature.size(0);
  const uint16 nChannel = feature.size(1);
  const uint16 nFeature = nChannel / nOrientation;

  auto mainDirection = at::empty({nBatch, nFeature}, feature.options().dtype(at::kByte));
  auto aligned = at::zeros_like(feature);
  
  AT_DISPATCH_FLOATING_TYPES(feature.type(), "RIE_forward", [&] {
    RIE_forward_cpu_kernel<scalar_t>(
         feature.data<scalar_t>(),
         mainDirection.data<uint8_t>(),
         aligned.data<scalar_t>(),
         nOrientation,
         nBatch,
         nFeature);
  });
  return std::make_tuple(mainDirection, aligned);
}


at::Tensor RIE_backward_cpu(const at::Tensor& mainDirection,
                            const at::Tensor& gradOutput,
                            const uint8 nOrientation) {
  AT_ASSERTM(!mainDirection.type().is_cuda(), "input must be a CPU tensor");
  AT_ASSERTM(!gradOutput.type().is_cuda(), "rois must be a CPU tensor");

  const uint16 nBatch = mainDirection.size(0);
  const uint16 nFeature = mainDirection.size(1);

  auto gradInput = at::zeros_like(gradOutput);

  AT_DISPATCH_FLOATING_TYPES(gradOutput.type(), "RIE_backward", [&] {
    RIE_backward_cpu_kernel<scalar_t>(
         mainDirection.data<uint8_t>(),
         gradOutput.data<scalar_t>(),
         nOrientation,
         nBatch,
         nFeature,
         gradInput.data<scalar_t>());
  });
  return gradInput;
}