#include "csrc/pybind.h"

#include <torch/extension.h>

#include "functions.h"
void init_perception_fused(pybind11::module& m)
{
    // bev_pool
    m.def("npu_bev_pool", &npu_bev_pool, "npu_bev_pool NPU version");
    m.def("npu_bev_pool_backward", &npu_bev_pool_backward, "npu_bev_pool_backward NPU version");
    m.def("npu_bev_pool_v2", &npu_bev_pool_v2, "npu_bev_pool_v2 NPU version");
    m.def("npu_bev_pool_v2_backward", &npu_bev_pool_v2_backward, "npu_bev_pool_v2_backward NPU version");
}
