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

#include <ATen/ATen.h>
#include "csrc/OpApiCommon.h"
#include "functions.h"

std::tuple<at::Tensor &, at::Tensor &> voxel_pooling_train(const at::Tensor& inputFeatures, const at::Tensor& geom,
    at::Tensor& outputFeatures, at::Tensor& posMemo, int batchSize, int numPoints, int numChannels,
    int numVoxelX, int numVoxelY, int numVoxelZ)
{
    at::Tensor geomTrans = geom.transpose(1, 2).contiguous();
    at::Tensor posMemoTrans = posMemo.transpose(1, 2).contiguous();
    EXEC_NPU_CMD(aclnnVoxelPoolingTrain, geomTrans, inputFeatures, batchSize, numPoints,
                 numChannels, numVoxelX, numVoxelY, numVoxelZ, outputFeatures, posMemoTrans);
    posMemo = posMemoTrans.transpose(1, 2).contiguous();
    return {posMemo, outputFeatures};
}