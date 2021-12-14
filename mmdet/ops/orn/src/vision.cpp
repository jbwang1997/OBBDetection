// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/extension.h>
#include "./ActiveRotatingFilter.h"
#include "./RotationInvariantEncoding.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("arf_forward", &ARF_forward, "ARF_forward");
  m.def("arf_backward", &ARF_backward, "ARF_backward");
  m.def("rie_forward", &RIE_forward, "RIE_forward");
  m.def("rie_backward", &RIE_backward, "RIE_backward");
}