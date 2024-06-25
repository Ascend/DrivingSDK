#include "csrc/pybind.h"

#include <torch/extension.h>

#include "functions.h"

void init_perception_point(pybind11::module& m)
{
    // group_points
    m.def("group_points", &group_points);
    m.def("group_points_backward", &group_points_backward);

    // vec_pool
    m.def("vec_pool_backward", &vec_pool_backward);

    m.def("point_to_voxel", &point_to_voxel);

    m.def("voxel_to_point", &voxel_to_point);

    m.def("unique_voxel", &unique_voxel);
}
