#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <torch/extension.h>

#define INF 10000000
#define EPS 0.000001

template <typename scalar_t>
at::Tensor convex_sort_cpu_kernel(
    const at::Tensor pts, const at::Tensor& masks, const bool circular) {
  auto nbs = pts.size(0);
  auto npts = pts.size(1);
  auto index_size = (circular) ? npts+1 : npts;
  if (nbs == 0) {
    return at::empty({nbs, index_size}, pts.options().dtype(at::kLong));
  }

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

  scalar_t* x = x_t.data_ptr<scalar_t>();
  scalar_t* y = y_t.data_ptr<scalar_t>();
  scalar_t* m = masks_t.data_ptr<scalar_t>();
  int64_t* start_index = start_index_t.data_ptr<int64_t>();
  int64_t* order = order_t.data_ptr<int64_t>();

  at::Tensor convex_index_t = at::full({nbs, index_size}, -1, pts.options().dtype(at::kLong));
  int64_t* convex_index = convex_index_t.data_ptr<int64_t>();

  for (int64_t i = 0; i < nbs; i++) {
    scalar_t* sub_x = x + i * npts;
    scalar_t* sub_y = y + i * npts;
    scalar_t* sub_m = m + i * npts;
    int64_t* sub_order = order + i * npts;
    int64_t* sub_convex_index = convex_index + i * index_size;
    int64_t sub_start_index = start_index[i];

    sub_convex_index[0] = sub_start_index;
    int64_t c_i = 0;

    for (int64_t _j = 0; _j < npts; _j++) {
      int64_t j = sub_order[_j];
      if (j == sub_start_index)continue;
      if (sub_m[j] < 0.5)continue;

      scalar_t x0 = sub_x[j];
      scalar_t y0 = sub_y[j];
      scalar_t x1 = sub_x[sub_convex_index[c_i]];
      scalar_t y1 = sub_y[sub_convex_index[c_i]];
      scalar_t d = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0);
      if (d < EPS)continue;

      if (c_i < 2) {
	sub_convex_index[++c_i] = j;
      }
      else {
	scalar_t x2 = sub_x[sub_convex_index[c_i - 1]];
	scalar_t y2 = sub_y[sub_convex_index[c_i - 1]];
        while(1) {
          scalar_t t = (x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2);
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
    if (circular)sub_convex_index[++c_i] = sub_convex_index[0];
  }
  return convex_index_t;
}

at::Tensor convex_sort_cpu(
    const at::Tensor& pts, const at::Tensor& masks, const bool circular) {
  AT_ASSERTM(pts.device().is_cpu(), "pts must be a CPU tensor");
  AT_ASSERTM(masks.device().is_cpu(), "masks must be a CPU tensor");
  AT_ASSERTM((pts.size(0) == masks.size(0)) && (pts.size(1) == masks.size(1)),
      "the first and second dimension of pts and masks must be in same size.");

  at::Tensor convex_index;
  AT_DISPATCH_FLOATING_TYPES(pts.scalar_type(), "convex_sort", [&] {
    convex_index = convex_sort_cpu_kernel<scalar_t>(pts, masks, circular);
  });

  return convex_index;
}
