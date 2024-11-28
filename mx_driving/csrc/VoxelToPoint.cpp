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

at::Tensor voxel_to_point(const at::Tensor& voxels, const std::vector<float> voxel_sizes,
    const std::vector<float> coor_ranges, const char* layout)
{
    TORCH_CHECK_NPU(voxels);
    TORCH_CHECK(voxels.dim() == 1, "voxels.dim() must be 1, but got: ", voxels.dim());

    at::Tensor points = at::empty({3, voxels.size(0)}, voxels.options().dtype(at::kInt));

    at::SmallVector<float, 3> voxel_sizes_vector {DEFAULT_VALUE, DEFAULT_VALUE, DEFAULT_VALUE};
    at::SmallVector<float, 6> coor_ranges_vector {
        DEFAULT_VALUE, DEFAULT_VALUE, DEFAULT_VALUE, DEFAULT_VALUE, DEFAULT_VALUE, DEFAULT_VALUE};
    at::ArrayRef<float> voxel_sizes_value;
    at::ArrayRef<float> coor_ranges_value;
    if (voxel_sizes.empty()) {
        voxel_sizes_value = at::ArrayRef<float>(voxel_sizes_vector);
    } else {
        TORCH_CHECK(voxel_sizes.size() == 3, "voxel_sizes.size() must be 3, but got: ", voxel_sizes.size());
        voxel_sizes_value = at::ArrayRef<float>(voxel_sizes);
    }
    if (coor_ranges.empty()) {
        coor_ranges_value = at::ArrayRef<float>(coor_ranges_vector);
    } else {
        TORCH_CHECK(coor_ranges.size() == 6, "coor_ranges.size() must be 6, but got: ", coor_ranges.size());
        coor_ranges_value = at::ArrayRef<float>(coor_ranges);
    }

    EXEC_NPU_CMD_SYNC(aclnnVoxelToPoint, voxels, voxel_sizes_value, coor_ranges_value, layout, points);
    return points.transpose(0, 1);
}
