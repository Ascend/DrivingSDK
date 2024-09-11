#include <torch/extension.h>
#include "csrc/pybind.h"
#include "functions.h"

void init_preprocess(pybind11::module& m)
{
    // npu_points_in_box
    m.def("npu_points_in_box", &npu_points_in_box);
    
    // npu_points_in_box_all
    m.def("npu_points_in_box_all", &npu_points_in_box_all);

    // npu_roipoint_pool3d_forward
    m.def("npu_roipoint_pool3d_forward", &npu_roipoint_pool3d_forward);
}
