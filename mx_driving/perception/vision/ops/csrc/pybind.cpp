#include <torch/extension.h>
#include "csrc/pybind.h"
#include "functions.h"

void init_perception_vision(pybind11::module &m)
{
    // npu_boxes_overlap_bev
    m.def("npu_boxes_overlap_bev", &npu_boxes_overlap_bev, "boxes_overlap_bev NPU version");
}
