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

at::Tensor point_to_voxel(const at::Tensor& points, const c10::optional<at::ArrayRef<float>> voxel_sizes,
    const c10::optional<at::ArrayRef<float>> coor_ranges);

at::Tensor voxel_to_point(const at::Tensor& voxels, const c10::optional<at::ArrayRef<float>> voxel_sizes,
    const c10::optional<at::ArrayRef<float>> coor_ranges);

std::tuple<int32_t, at::Tensor, at::Tensor, at::Tensor> unique_voxel(const at::Tensor& voxels);


#endif // PERCEPTION_POINT_OPS_CSRC_FUNCTIONS_H_
