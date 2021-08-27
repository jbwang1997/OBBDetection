#include <ATen/ATen.h>
#include <torch/extension.h>

#ifdef WITH_CUDA
at::Tensor convex_sort_cuda(
    const at::Tensor& pts, const at::Tensor& masks, const bool circular);
#endif

at::Tensor convex_sort_cpu(
    const at::Tensor& pts, const at::Tensor& masks, const bool circular);


at::Tensor convex_sort(
    const at::Tensor& pts, const at::Tensor& masks, const bool circular) {
  if (pts.device().is_cuda()) {
#ifdef WITH_CUDA
    return convex_sort_cuda(pts, masks, circular);
#else
    AT_ERROR("sort_vert is not compiled with GPU support");
#endif
  }
  return convex_sort_cpu(pts, masks, circular);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("convex_sort", &convex_sort, "select the convex points and sort them");
}
