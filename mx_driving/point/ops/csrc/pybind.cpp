#include "csrc/pybind.h"

#include <torch/extension.h>

#include "functions.h"

void init_point(pybind11::module& m)
{
    // group_points
    m.def("group_points", &group_points);
    m.def("group_points_backward", &group_points_backward);

    // vec_pool
    m.def("vec_pool_backward", &vec_pool_backward);

    m.def("point_to_voxel", &point_to_voxel);

    m.def("voxel_to_point", &voxel_to_point);

    m.def("unique_voxel", &unique_voxel);
    
    m.def("hard_voxelize", &hard_voxelize);

    // bev_pool
    m.def("npu_bev_pool", &npu_bev_pool, "npu_bev_pool NPU version");
    m.def("npu_bev_pool_backward", &npu_bev_pool_backward, "npu_bev_pool_backward NPU version");
    m.def("npu_bev_pool_v2", &npu_bev_pool_v2, "npu_bev_pool_v2 NPU version");
    m.def("npu_bev_pool_v2_backward", &npu_bev_pool_v2_backward, "npu_bev_pool_v2_backward NPU version");

    // furthest_points_sampling_with_dist
    m.def("furthest_point_sampling_with_dist", &furthest_point_sampling_with_dist);

    // npu_dynamic_scatter
    m.def("npu_dynamic_scatter", &npu_dynamic_scatter);
    m.def("npu_dynamic_scatter_grad", &npu_dynamic_scatter_grad);

    // dyn_voxelization
    m.def("dynamic_voxelization", &dynamic_voxelization);

    // npu_furthest_point_sampling
    m.def("npu_furthest_point_sampling", &npu_furthest_point_sampling);

    // voxel_pooling
    m.def("voxel_pooling_train", &voxel_pooling_train);
    m.def("voxel_pool_train_backward", &voxel_pool_train_backward);
}
