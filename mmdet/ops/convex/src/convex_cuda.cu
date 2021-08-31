#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#define INF 10000000
#define EPS 0.000001

template <typename T>
__global__ void convex_sort_kernel(
    const int nbs, const int npts, const int index_size, const bool circular,
    const T* x, const T* y, const T* m, const int64_t* start_index,
    const int64_t* order, int64_t* convex_index) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nbs) {
    const T* sub_x = x + i * npts;
    const T* sub_y = y + i * npts;
    const T* sub_m = m + i * npts;
    const int64_t* sub_order = order + i * npts;
    const int64_t sub_start_index = start_index[i];

    int64_t* sub_convex_index = convex_index + i * index_size;
    sub_convex_index[0] = sub_start_index;
    int64_t c_i = 0;

    for (int _j = 0; _j < npts; _j++) {
      const int64_t j = sub_order[_j];
      if (j == sub_start_index)continue;
      if (sub_m[j] < 0.5)continue;

      const T x0 = sub_x[j];
      const T y0 = sub_y[j];
      T x1 = sub_x[sub_convex_index[c_i]];
      T y1 = sub_y[sub_convex_index[c_i]];
      T d = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0);
      if (d < EPS)continue;

      if (c_i < 2) {
	sub_convex_index[++c_i] = j;
      }
      else {
	T x2 = sub_x[sub_convex_index[c_i - 1]];
	T y2 = sub_y[sub_convex_index[c_i - 1]];
	while(1) {
	  T t = (x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2);
	  if (t >= 0) {
	    sub_convex_index[++c_i] = j;
	    break;
	  }
	  else {
	    if (c_i <= 1) {
	      sub_convex_index[c_i] = j;
	      break;
	    }
	    else {
	      c_i--;
	      x1 = sub_x[sub_convex_index[c_i]];
	      y1 = sub_y[sub_convex_index[c_i]];
	      x2 = sub_x[sub_convex_index[c_i - 1]];
	      y2 = sub_y[sub_convex_index[c_i - 1]];
	    }
	  }
	}
      }
    }
    if (circular) sub_convex_index[++c_i] = sub_convex_index[0];
  }
}


at::Tensor convex_sort_cuda(
    const at::Tensor& pts, const at::Tensor& masks, const bool circular) {
  AT_ASSERTM(pts.device().is_cuda(), "pts must be a CUDA tensor");
  AT_ASSERTM(masks.device().is_cuda(), "masks must be a CUDA tensor");
  AT_ASSERTM((pts.size(0) == masks.size(0)) && (pts.size(1) == masks.size(1)),
      "the first and second dimension of pts and masks must be in same size.");

  at::CheckedFrom c = "sort_vert_cuda";
  at::TensorArg pts_arg{pts, "pts", 1}, masks_arg{masks, "masks", 2};
  at::checkAllSameGPU(c, {pts_arg, masks_arg});
  at::cuda::CUDAGuard device_guard(pts.device());

  int nbs = pts.size(0);
  int npts = pts.size(1);
  int index_size = (circular) ? npts+1 : npts;

  auto masks_t = masks.toType(pts.scalar_type()).contiguous();
  auto x_t = pts.select(2, 0).contiguous();
  auto y_t = pts.select(2, 1).contiguous();

  auto masked_y = masks_t * y_t + (1 - masks_t) * INF;
  auto start_index_t = masked_y.argmin(1, /*keepdim*/true);
  auto start_x = x_t.gather(1, start_index_t);
  auto start_y = y_t.gather(1, start_index_t);

  auto pts_cos = (x_t - start_x) / torch::sqrt(
      (x_t - start_x)*(x_t - start_x) + (y_t - start_y)*(y_t - start_y) + EPS);
  auto order_t = pts_cos.argsort(1, /*descend*/true);

  at::Tensor convex_index_t = at::full({nbs, index_size}, -1, pts.options().dtype(at::kLong));
  if (npts == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return convex_index_t;
  }

  dim3 blocks(THCCeilDiv(nbs, 512));
  dim3 threads(512);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(pts.scalar_type(), "convex_sort", [&] {
      convex_sort_kernel<<<blocks, threads, 0, stream>>>(
	  nbs, npts, index_size, circular, x_t.data_ptr<scalar_t>(),
	  y_t.data_ptr<scalar_t>(), masks_t.data_ptr<scalar_t>(),
	  start_index_t.data_ptr<int64_t>(), order_t.data_ptr<int64_t>(),
	  convex_index_t.data_ptr<int64_t>());
  });
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());
  return convex_index_t;
}
