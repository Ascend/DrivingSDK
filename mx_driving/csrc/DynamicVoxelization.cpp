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

at::Tensor dynamic_voxelization(const at::Tensor& points, at::Tensor& coors, int grid_x, int grid_y, int grid_z,
    double voxel_x, double voxel_y, double voxel_z, double coors_min_x, double coors_min_y, double coorsMinZ)
{
    uint32_t ptsNum = points.size(0);
    uint32_t ptsFeature = points.size(1);
    at::Tensor ptsTrans = at::transpose(points, 0, 1);
    EXEC_NPU_CMD(aclnnDynamicVoxelization, ptsTrans, coors_min_x, coors_min_y, coorsMinZ, voxel_x, voxel_y, voxel_z,
        grid_x, grid_y, grid_z, coors);
    coors.transpose_(0, 1);
    return coors;
}
