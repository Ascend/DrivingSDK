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

#ifndef PERCEPTION_POINT_OPS_CSRC_FUNCTIONS_H_
#define PERCEPTION_POINT_OPS_CSRC_FUNCTIONS_H_
#include <ATen/Tensor.h>
#include <torch/library.h>

#include <tuple>

at::Tensor group_points(
    const at::Tensor& points, const at::Tensor& idx, int64_t b, int64_t c, int64_t n, int64_t npoints, int64_t nsample);

at::Tensor group_points_backward(const at::Tensor& grad_out, const at::Tensor& idx, int64_t b, int64_t c, int64_t n,
    int64_t npoints, int64_t nsample);

at::Tensor vec_pool_backward(const at::Tensor& grad_new_features, const at::Tensor& point_cnt_of_grid,
    const at::Tensor& grouped_idxs, const int64_t n, const int64_t num_c_in);

at::Tensor point_to_voxel(
    const at::Tensor& points, const std::vector<float> voxel_sizes, const std::vector<float> coor_ranges);

at::Tensor voxel_to_point(
    const at::Tensor& voxels, const std::vector<float> voxel_sizes, const std::vector<float> coor_ranges);

std::tuple<int32_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor> unique_voxel(const at::Tensor& voxels);

std::tuple<int32_t, at::Tensor, at::Tensor, at::Tensor> hard_voxelize(const at::Tensor& points,
    const std::vector<float> voxel_sizes, const std::vector<float> coor_ranges, int64_t max_points, int64_t max_voxels);

at::Tensor npu_bev_pool(const at::Tensor& feat, const at::Tensor& geom_feat, const at::Tensor& interval_lengths,
    const at::Tensor& interval_starts, int64_t b, int64_t d, int64_t h, int64_t w);
at::Tensor npu_bev_pool_backward(const at::Tensor& grad_out, const at::Tensor& geom_feat,
    const at::Tensor& interval_lengths, const at::Tensor& interval_starts, int64_t b, int64_t d, int64_t h, int64_t w);

at::Tensor npu_bev_pool_v2(const at::Tensor& depth, const at::Tensor& feat, const at::Tensor& ranks_depth,
    const at::Tensor& ranks_feat, const at::Tensor& ranks_bev, const at::Tensor& interval_lengths,
    const at::Tensor& interval_starts, int64_t b, int64_t d, int64_t h, int64_t w);
std::tuple<at::Tensor, at::Tensor> npu_bev_pool_v2_backward(const at::Tensor& grad_out, const at::Tensor& depth,
    const at::Tensor& feat, const at::Tensor& ranks_depth, const at::Tensor& ranks_feat, const at::Tensor& ranks_bev,
    const at::Tensor& interval_lengths, const at::Tensor& interval_starts, int64_t b, int64_t d, int64_t h, int64_t w);

at::Tensor furthest_point_sampling_with_dist(
    const at::Tensor& points_dist, const at::Tensor& nearest_temp, int32_t num_points);

std::tuple<at::Tensor, at::Tensor> npu_dynamic_scatter(const at::Tensor& feats, const at::Tensor& coors,
    const at::Tensor& prefix_sum_point_per_voxel, const at::Tensor& argsort_coor, int32_t num_voxels,
    const char* reduce_type);

void npu_dynamic_scatter_grad(at::Tensor& grad_point_feats, const at::Tensor& grad_voxel_feats,
    const at::Tensor& prefix_sum_point_per_voxel, const at::Tensor& argsort_coor, const at::Tensor& compare_mask,
    const char* reduce_type);

at::Tensor npu_furthest_point_sampling(const at::Tensor& point_xyz, const at::Tensor& nearset_temp, int32_t num_points);

std::tuple<at::Tensor&, at::Tensor&> voxel_pooling_train(const at::Tensor& inputFeatures, const at::Tensor& geom,
    at::Tensor& outputFeatures, at::Tensor& posMemo, int batchSize, int numPoints, int numChannels, int numVoxelX,
    int numVoxelY, int numVoxelZ);

at::Tensor voxel_pool_train_backward(const at::Tensor& grad_out, const at::Tensor& posMemo, const int64_t batchSize,
    const int64_t numPoints, const int64_t numChannels, const int64_t h, const int64_t w);

at::Tensor dynamic_voxelization(const at::Tensor& points, at::Tensor& coors, int grid_x, int grid_y, int grid_z,
    double voxel_x, double voxel_y, double voxel_z, double coors_min_x, double coors_min_y, double coorsMinZ);

#endif // PERCEPTION_POINT_OPS_CSRC_FUNCTIONS_H_
