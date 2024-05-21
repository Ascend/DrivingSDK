// Copyright (c) 2024 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "csrc/pybind.h"

#include <torch/extension.h>

#include "functions.h"

void init_common(pybind11::module& m)
{
    // three_interpolate
    m.def("npu_three_interpolate", &npu_three_interpolate);
    m.def("npu_three_interpolate_backward", &npu_three_interpolate_backward);
    
    // scatter_max
    m.def("npu_scatter_max", &npu_scatter_max);
    m.def("npu_scatter_max_backward", &npu_scatter_max_backward);

    // roated overlap
    m.def("npu_rotated_overlaps", &npu_rotated_overlaps, "npu_rotated_overlap NPU version");

    // rotated iou
    m.def("npu_rotated_iou", &npu_rotated_iou);

    // furthest_points_sampling_with_dist
    m.def("furthest_point_sampling_with_dist", &furthest_point_sampling_with_dist);

    // npu_points_in_box
    m.def("npu_points_in_box", &npu_points_in_box);

    // npu_multi_scale_deformable_attn_function
    m.def("npu_multi_scale_deformable_attn_function", &npu_multi_scale_deformable_attn_function);
    m.def("multi_scale_deformable_attn_grad", &multi_scale_deformable_attn_grad);

    // npu_dynamic_scatter
    m.def("npu_dynamic_scatter", &npu_dynamic_scatter);

    // dyn_voxelization
    m.def("dynamic_voxelization", &dynamic_voxelization);

    // nms3d_normal
    m.def("nms3d_normal", &nms3d_normal);

    // nms3d
    m.def("nms3d", &nms3d);

    // npu_furthest_point_sampling
    m.def("npu_furthest_point_sampling", &npu_furthest_point_sampling);

    // npu_scatter_mean_grad
    m.def("npu_scatter_mean_grad", &npu_scatter_mean_grad);

    // voxel_pooling
    m.def("voxel_pooling_train", &voxel_pooling_train);
}
