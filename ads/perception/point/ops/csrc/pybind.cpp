#include "csrc/pybind.h"

#include <torch/extension.h>

#include "functions.h"

void init_perception_point(pybind11::module& m)
{
    // group_points
    m.def("group_points", &group_points);

    // vec_pool
    m.def("vec_pool_backward", &vec_pool_backward);
}