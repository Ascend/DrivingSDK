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

#include "csrc/OpApiCommon.h"
#include "csrc/functions.h"

constexpr size_t NUM_VOXELS_IDX = 0;
constexpr size_t UNI_VOXELS_IDX = 1;
constexpr size_t UNI_INDICES_IDX = 2;
constexpr size_t ARGSORT_INDICES_IDX = 3;
constexpr size_t UNI_ARGSORT_INDICES_IDX = 4;
constexpr size_t POINT_NUM_DIM = 0;
constexpr size_t FEAT_NUM_DIM = 1;


std::tuple<int32_t, at::Tensor, at::Tensor, at::Tensor> hard_voxelize(const at::Tensor& points,
    const std::vector<float> voxel_sizes, const std::vector<float> coor_ranges, int64_t max_points, int64_t max_voxels)
{
    TORCH_CHECK_NPU(points);
    TORCH_CHECK(points.dim() == 2, "points.dim() must be 2, but got: ", points.dim());
    size_t point_num = points.size(POINT_NUM_DIM);
    size_t feat_num = points.size(FEAT_NUM_DIM);
    auto voxels = point_to_voxel(points, voxel_sizes, coor_ranges, "XYZ");
    auto uni_res = unique_voxel(voxels);
    int32_t num_voxels = std::get<NUM_VOXELS_IDX>(uni_res);
    at::Tensor uni_voxels = std::get<UNI_VOXELS_IDX>(uni_res);
    at::Tensor uni_indices = std::get<UNI_INDICES_IDX>(uni_res);
    at::Tensor argsort_indices = std::get<ARGSORT_INDICES_IDX>(uni_res);
    at::Tensor uni_argsort_indices = std::get<UNI_ARGSORT_INDICES_IDX>(uni_res);

    auto sorted = at::sort(uni_argsort_indices);
    at::Tensor uni_argsort_argsort_indices = std::get<1>(sorted).to(at::kInt);

    int64_t real_num_voxels = std::min(static_cast<int64_t>(num_voxels), max_voxels);
    at::Tensor vox_points = at::zeros({real_num_voxels, max_points, feat_num}, points.options());
    at::Tensor num_points_per_voxel = at::empty({real_num_voxels}, points.options().dtype(at::kInt));
    at::Tensor sorted_uni_voxels = at::empty({real_num_voxels}, voxels.options().dtype(at::kInt));

    EXEC_NPU_CMD(aclnnHardVoxelize, points, uni_voxels, argsort_indices, uni_argsort_argsort_indices, uni_indices,
        num_voxels, max_voxels, max_points, point_num, vox_points, num_points_per_voxel, sorted_uni_voxels);

    at::Tensor coors = voxel_to_point(sorted_uni_voxels, voxel_sizes, coor_ranges, "XYZ");
    return std::tie(real_num_voxels, vox_points, coors, num_points_per_voxel);
}
