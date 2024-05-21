// Copyright (c) 2024, Huawei Technologies.All rights reserved.
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
#ifndef COMMON_OPS_CSRC_FUNCTIONS_H_
#define COMMON_OPS_CSRC_FUNCTIONS_H_

#include <ATen/ATen.h>

at::Tensor npu_three_interpolate(int b, int c, int m, int n, const at::Tensor& points, const at::Tensor& idx, const at::Tensor& weight);
at::Tensor npu_three_interpolate_backward(int b, int c, int n, int m, const at::Tensor& grad_out, const at::Tensor& idx, const at::Tensor& weight);

std::tuple<at::Tensor, at::Tensor> npu_scatter_max(
    const at::Tensor& updates, const at::Tensor& indices, c10::optional<at::Tensor> out);
at::Tensor npu_scatter_max_backward(const at::Tensor& x, const at::Tensor& segment_ids, const at::Tensor& num_segments);
at::Tensor npu_rotated_overlaps(const at::Tensor& self, const at::Tensor& query_boxes, bool trans);
at::Tensor npu_rotated_iou(const at::Tensor& boxes, const at::Tensor& query_boxes, bool trans, int64_t mode,
    bool is_cross, double v_threshold, double e_threshold);
at::Tensor npu_scatter(const at::Tensor& self, const at::Tensor& indices, const at::Tensor& updates, int64_t dim);

at::Tensor furthest_point_sampling_with_dist(
    const at::Tensor& points_dist, const at::Tensor& nearest_temp, int32_t num_points);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_dynamic_scatter(
    at::Tensor& cof_tensor, at::Tensor& out_coors_unique2, at::Tensor& coors_map,
    at::Tensor& reduce_count, const at::Tensor& feats, const at::Tensor& coors, int64_t reduce_type);

at::Tensor npu_points_in_box(const at::Tensor& boxes, const at::Tensor& pts);
at::Tensor npu_multi_scale_deformable_attn_function(const at::Tensor& value, const at::Tensor& value_spatial_shapes,
    const at::Tensor& value_level_start_index, const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights);
std::tuple<at::Tensor, at::Tensor, at::Tensor> multi_scale_deformable_attn_grad(const at::Tensor& value,
    const at::Tensor& shape, const at::Tensor& level_start_index, const at::Tensor& location,
    const at::Tensor& attn_weight, const at::Tensor& grad_output);
at::Tensor npu_furthest_point_sampling(const at::Tensor& point_xyz, const at::Tensor& nearset_temp, int32_t num_points);
at::Tensor dynamic_voxelization(const at::Tensor& points, at::Tensor& coors, int grid_x, int grid_y, int grid_z,
    double voxel_x, double voxel_y, double voxel_z, double coors_min_x, double coors_min_y, double coorsMinZ);

std::tuple<at::Tensor, at::Tensor> nms3d_normal(const at::Tensor& boxes, double nms_overlap_thresh);

std::tuple<at::Tensor, at::Tensor> nms3d(const at::Tensor& boxes, double threshold);
at::Tensor npu_scatter_mean_grad(const at::Tensor &grad_out, const at::Tensor &index, int32_t dim);
std::tuple<at::Tensor &, at::Tensor &> voxel_pooling_train(const at::Tensor& inputFeatures, const at::Tensor& geom,
    at::Tensor& outputFeatures, at::Tensor& posMemo, int batchSize, int numPoints, int numChannels,
    int numVoxelX, int numVoxelY, int numVoxelZ);
#endif // COMMON_OPS_CSRC_FUNCTIONS_H_
