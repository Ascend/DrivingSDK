// Copyright (c) 2024-2025 Huawei Technologies Co., Ltd
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

#include "csrc/functions.h"
#include <torch/extension.h>

#include <mutex>
#include <string>

std::string g_opApiSoPath;
std::once_flag init_flag; // Flag for one-time initialization

void init_op_api_so_path(const std::string& path)
{
    std::call_once(init_flag, [&]() { g_opApiSoPath = path; });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("_init_op_api_so_path", &init_op_api_so_path);
    // knn
    m.def("knn", &knn);

    // npu_scatter_mean_grad
    m.def("npu_scatter_mean_grad", &npu_scatter_mean_grad);

    // three_interpolate
    m.def("npu_three_interpolate", &npu_three_interpolate);
    m.def("npu_three_interpolate_backward", &npu_three_interpolate_backward);

    // scatter_mean
    m.def("npu_scatter_mean", &npu_scatter_mean, "npu_scatter_mean NPU version");

    // scatter_max
    m.def("scatter_max_with_argmax_v2", &scatter_max_with_argmax_v2);
    m.def("npu_scatter_max_backward", &npu_scatter_max_backward);

    // npu_sort_pairs
    m.def("npu_sort_pairs", &npu_sort_pairs, "sort_pairs NPU version");

    // npu_hypot
    m.def("npu_hypot", &npu_hypot);
    m.def("npu_hypot_grad", &npu_hypot_grad);

    // assign_score_withk
    m.def("assign_score_withk", &assign_score_withk);
    m.def("assign_score_withk_grad", &assign_score_withk_grad);
    // nms3d_normal
    m.def("nms3d_normal", &nms3d_normal);

    // nms3d
    m.def("nms3d", &nms3d);

    // nms3d_on_sight
    m.def("nms3d_on_sight", &nms3d_on_sight);

    // roated overlap
    m.def("npu_rotated_overlaps", &npu_rotated_overlaps, "npu_rotated_overlap NPU version");

    // rotated iou
    m.def("npu_rotated_iou", &npu_rotated_iou);

    // npu_boxes_overlap_bev
    m.def("npu_boxes_overlap_bev", &npu_boxes_overlap_bev, "boxes_overlap_bev NPU version");

    // roi_align_rotated_v2_forward_npu
    m.def("roi_align_rotated_v2_forward_npu", &roi_align_rotated_v2_forward_npu);

    // npu_roi_align_rotated_grad_v2
    m.def("npu_roi_align_rotated_grad_v2", &npu_roi_align_rotated_grad_v2);

    // npu_box_iou_quadri
    m.def("npu_box_iou_quadri", &npu_box_iou_quadri, "box_iou_quadri NPU version");

    // npu_box_iou_rotated
    m.def("npu_box_iou_rotated", &npu_box_iou_rotated, "box_iou_rotated NPU version");

    // border_align_forward_npu
    m.def("border_align", &border_align);

    // border_align_backward_npu
    m.def("border_align_backward", &border_align_backward);

    // npu_roiaware_pool3d_forward
    m.def("npu_roiaware_pool3d_forward", &npu_roiaware_pool3d_forward);

    // roiaware_pool3d_grad
    m.def("roiaware_pool3d_grad", &roiaware_pool3d_grad, "roiaware_pool3d_grad NPU version");

    // pixel_group
    m.def("pixel_group", &pixel_group);

    // nnpu_max_pool2d
    m.def("npu_max_pool2d", &npu_max_pool2d);
    // mullti_scale_deformable_attn
    m.def("multi_scale_deformable_attn", &multi_scale_deformable_attn);
    m.def("multi_scale_deformable_attn_backward", &multi_scale_deformable_attn_backward);

    // npu_add_relu
    m.def("npu_add_relu", &npu_add_relu);
    m.def("npu_add_relu_grad", &npu_add_relu_grad);

    // fused_bias_leaky_relu
    m.def("fused_bias_leaky_relu", &fused_bias_leaky_relu);

    // npu_deformable_aggregation
    m.def("npu_deformable_aggregation", &deformable_aggregation);
    m.def("npu_deformable_aggregation_backward", &deformable_aggregation_backward);

    // deformable_conv2d
    m.def("deformable_conv2d", &deformable_conv2d);
    m.def("modulated_deformable_conv2d", &modulated_deformable_conv2d);
    m.def("deformable_conv2d_backward", &deformable_conv2d_backward);
    m.def("modulated_deformable_conv2d_backward", &modulated_deformable_conv2d_backward);

    // npu_geometric_kernel_attention_func
    m.def("geometric_kernel_attention_forward", &geometric_kernel_attention_forward);
    m.def("geometric_kernel_attention_backward", &geometric_kernel_attention_backward);

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
    m.def("npu_bev_pool_v3", &npu_bev_pool_v3, "npu_bev_pool_v3 NPU version");
    m.def("npu_bev_pool_v3_backward", &npu_bev_pool_v3_backward, "npu_bev_pool_v3_backward NPU version");

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

    // npu_points_in_box
    m.def("npu_points_in_box", &npu_points_in_box);

    // npu_points_in_box_all
    m.def("npu_points_in_box_all", &npu_points_in_box_all);

    // npu_roipoint_pool3d_forward
    m.def("npu_roipoint_pool3d_forward", &npu_roipoint_pool3d_forward);

    // npu_subm_sparse_conv3d
    m.def("npu_subm_sparse_conv3d", &npu_subm_sparse_conv3d);

    // npu_sparse_conv3d
    m.def("npu_sparse_conv3d", &npu_sparse_conv3d);

    // multi_to_sparse
    m.def("multi_to_sparse", &multi_to_sparse);

    // multi_to_sparse_v2
    m.def("multi_to_sparse_v2", &multi_to_sparse_v2);

    // npu_sparse_conv3d_grad
    m.def("npu_sparse_conv3d_grad", &npu_sparse_conv3d_grad);

    // npu_prepare_subm_conv3d
    m.def("npu_prepare_subm_conv3d", &npu_prepare_subm_conv3d);

    // cal_anchors_heading
    m.def("cal_anchors_heading", &cal_anchors_heading);
    // npu_subm_sparse_conv3d_grad
    m.def("npu_subm_sparse_conv3d_grad", &npu_subm_sparse_conv3d_grad);

    // npu_gaussian
    m.def("npu_gaussian", &npu_gaussian);

    // npu_draw_gaussian_to_heatmap
    m.def("npu_draw_gaussian_to_heatmap", &npu_draw_gaussian_to_heatmap);

    // npu_assign_target_of_single_head
    m.def("npu_assign_target_of_single_head", &npu_assign_target_of_single_head);

    // diff_iou_rotated_sort_vertices
    m.def("diff_iou_rotated_sort_vertices", &diff_iou_rotated_sort_vertices);

    // grid_sampler2d_v2
    m.def("grid_sampler2d_v2", &grid_sampler2d_v2);

    // grid_sampler2d_v2_backward
    m.def("grid_sampler2d_v2_backward", &grid_sampler2d_v2_backward);

    // scatter_add
    m.def("npu_scatter_add", &npu_scatter_add);

    // scatter_add_grad
    m.def("npu_scatter_add_grad", &npu_scatter_add_grad);

    // select_idx_with_mask
    m.def("select_idx_with_mask", &select_idx_with_mask);

    // cartesian_to_frenet1
    m.def("cartesian_to_frenet1", &cartesian_to_frenet1);

    // calc_poly_start_end_sl
    m.def("calc_poly_start_end_sl", &calc_poly_start_end_sl);

    // npu_batch_matmul
    m.def("npu_batch_matmul", &npu_batch_matmul);

    // npu_subm_sparse_conv3d_with_key
    m.def("npu_subm_sparse_conv3d_with_key", &npu_subm_sparse_conv3d_with_key);

    // min_area_polygons
    m.def("min_area_polygons", &min_area_polygons);

    // npu_subm_sparse_conv3d_v2
    m.def("npu_subm_sparse_conv3d_v2", &npu_subm_sparse_conv3d_v2);

    // radius
    m.def("radius", &radius);
}
