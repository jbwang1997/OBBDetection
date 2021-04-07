#include <ATen/ATen.h>
#include <torch/extension.h>


#ifdef WITH_CUDA
at::Tensor ROIAlignRotated_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio);

at::Tensor ROIAlignRotated_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio);
#endif

at::Tensor ROIAlignRotated_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio);

at::Tensor ROIAlignRotated_backward_cpu(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio);


inline at::Tensor ROIAlignRotated_forward(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio) {
  if (input.device().is_cuda()) {
#ifdef WITH_CUDA
    return ROIAlignRotated_forward_cuda(
		    input, 
		    rois, 
		    spatial_scale, 
		    pooled_height,
		    pooled_width,
		    sampling_ratio);
#else
    AT_ERROR("ROIAlignRotated is not compiled with GPU support");
#endif
  }
  return ROIAlignRotated_forward_cpu(
		  input,
		  rois,
		  spatial_scale,
		  pooled_height,
		  pooled_width,
		  sampling_ratio);
}

inline at::Tensor ROIAlignRotated_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio) {
  if (grad.device().is_cuda()) {
#ifdef WITH_CUDA
    return ROIAlignRotated_backward_cuda(
		    grad,
		    rois,
		    spatial_scale,
		    pooled_height,
		    pooled_width,
		    batch_size,
		    channels,
		    height,
		    width,
		    sampling_ratio);
#else
    AT_ERROR("ROIAlignRotated is not compoled with GPU support");
#endif
  }
  return ROIAlignRotated_backward_cpu(
		  grad,
		  rois,
		  spatial_scale,
		  pooled_height,
		  pooled_width,
		  batch_size,
		  channels,
		  height,
		  width,
		  sampling_ratio);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ROIAlignRotated_forward, "Roi_Align_Rotated forward");
  m.def("backward", &ROIAlignRotated_backward, "Roi_Align_Rotated_backward");
}
