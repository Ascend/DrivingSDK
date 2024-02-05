// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include <ATen/ATen.h>
#include "csrc/OpApiCommon.h"
#include "functions.h"

at::Tensor DynamicVoxelization(
    const at::Tensor &points,
    at::Tensor &coors,
    const int gridX,
    const int gridY,
    const int gridZ,
    const double voxelX,
    const double voxelY,
    const double voxelZ,
    const double coorsMinX,
    const double coorsMinY,
    const double coorsMinZ)
{
    uint32_t ptsNum = points.size(0);
    uint32_t ptsFeature = points.size(1);
    at::Tensor pts = at::transpose(points, 0, 1);
    at::Tensor ptsTrans = at::reshape(pts, {ptsNum, ptsFeature});
    EXEC_NPU_CMD(aclnnDynamicVoxelization, ptsTrans, coorsMinX, coorsMinY,
                 coorsMinZ, voxelX, voxelY, voxelZ, gridX, gridY, gridZ, coors);
    coors.transpose_(0, 1);
    return coors;
}
