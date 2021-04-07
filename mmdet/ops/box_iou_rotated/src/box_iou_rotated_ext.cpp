#include <ATen/ATen.h>
#include <torch/extension.h>

#ifdef WITH_CUDA
at::Tensor box_iou_rotated_cuda(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2,
    const bool iou_or_iof);
#endif

at::Tensor box_iou_rotated_cpu(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2,
    const bool iou_or_iof);


inline at::Tensor box_iou_rotated(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2,
    const bool iou_or_iof) {
  assert(boxes1.device().is_cuda() == boxes2.device().is_cuda());
  if (boxes1.device().is_cuda()) {
#ifdef WITH_CUDA
    return box_iou_rotated_cuda(
        boxes1.contiguous(),
	boxes2.contiguous(),
	iou_or_iof);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  return box_iou_rotated_cpu(
      boxes1.contiguous(),
      boxes2.contiguous(),
      iou_or_iof);
}
  
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("overlaps", box_iou_rotated, "calculate iou or iof of two group boxes");
}
