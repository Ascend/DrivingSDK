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

constexpr float DEFAULT_VALUE = -1.0f;
constexpr size_t VOXEL_SIZES_SIZE = 3;
constexpr size_t COOR_RANGES_SIZE = 6;

at::Tensor point_to_voxel(const at::Tensor& points, const std::vector<float> voxel_sizes,
    const std::vector<float> coor_ranges, const char* layout)
{
    TORCH_CHECK_NPU(points);
    TORCH_CHECK(points.dim() == 2, "points.dim() must be 2, but got: ", points.dim());

    at::Tensor voxels = at::empty({points.size(0)}, points.options().dtype(at::kFloat));

    at::SmallVector<float, VOXEL_SIZES_SIZE> voxel_sizes_vector {DEFAULT_VALUE, DEFAULT_VALUE, DEFAULT_VALUE};
    at::SmallVector<float, COOR_RANGES_SIZE> coor_ranges_vector {
        DEFAULT_VALUE, DEFAULT_VALUE, DEFAULT_VALUE, DEFAULT_VALUE, DEFAULT_VALUE, DEFAULT_VALUE};
    at::ArrayRef<float> voxel_sizes_value;
    at::ArrayRef<float> coor_ranges_value;
    if (voxel_sizes.empty()) {
        voxel_sizes_value = at::ArrayRef<float>(voxel_sizes_vector);
    } else {
        TORCH_CHECK(
            voxel_sizes.size() == VOXEL_SIZES_SIZE, "voxel_sizes.size() must be 3, but got: ", voxel_sizes.size());
        voxel_sizes_value = at::ArrayRef<float>(voxel_sizes);
    }
    if (coor_ranges.empty()) {
        coor_ranges_value = at::ArrayRef<float>(coor_ranges_vector);
    } else {
        TORCH_CHECK(
            coor_ranges.size() == COOR_RANGES_SIZE, "coor_ranges.size() must be 6, but got: ", coor_ranges.size());
        coor_ranges_value = at::ArrayRef<float>(coor_ranges);
    }
    // transpose points
    at::Tensor points_transpose = points.transpose(0, 1);
    EXEC_NPU_CMD(aclnnPointToVoxel, points_transpose, voxel_sizes_value, coor_ranges_value, layout, voxels);
    return voxels;
}
