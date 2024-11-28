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

std::tuple<at::Tensor&, at::Tensor&> voxel_pooling_train(const at::Tensor& inputFeatures, const at::Tensor& geom,
    at::Tensor& outputFeatures, at::Tensor& posMemo, int batchSize, int numPoints, int numChannels, int numVoxelX,
    int numVoxelY, int numVoxelZ)
{
    TORCH_CHECK_NPU(inputFeatures);
    TORCH_CHECK_NPU(geom);
    TORCH_CHECK(inputFeatures.dim() == 3, "inputFeatures.dim() must be 3, but got: ", inputFeatures.dim());
    TORCH_CHECK(geom.dim() == 3, "geom.dim() must be 3, but got: ", geom.dim());

    auto origin_dtype = inputFeatures.dtype();

    at::Tensor inputFeatures_cast = inputFeatures;
    if (origin_dtype == at::kHalf) {
        inputFeatures_cast = inputFeatures.to(at::kFloat);
        outputFeatures = outputFeatures.to(at::kFloat);
    }

    EXEC_NPU_CMD(aclnnVoxelPoolingTrain, geom, inputFeatures_cast, batchSize, numPoints, numChannels, numVoxelX,
        numVoxelY, numVoxelZ, outputFeatures, posMemo);

    if (origin_dtype == at::kHalf) {
        outputFeatures = outputFeatures.to(at::kHalf);
    }

    return {posMemo, outputFeatures};
}

at::Tensor voxel_pool_train_backward(const at::Tensor& gradOut, const at::Tensor& posMemo, const int64_t batchSize,
    const int64_t numPoints, const int64_t numChannels, const int64_t h, const int64_t w)
{
    TORCH_CHECK_NPU(gradOut);
    TORCH_CHECK_NPU(posMemo);
    TORCH_CHECK(gradOut.dim() == 4, "gradOut.dim() must be 4, but got: ", gradOut.dim());
    TORCH_CHECK(posMemo.dim() == 3, "posMemo.dim() must be 3, but got: ", posMemo.dim());

    auto origin_dtype = gradOut.dtype();

    at::Tensor gradOutTensor = gradOut.permute({0, 2, 3, 1}).contiguous();
    at::Tensor out = at::zeros({batchSize, numPoints, numChannels}, gradOut.options());

    if (origin_dtype == at::kHalf) {
        out = out.to(at::kFloat);
        gradOutTensor = gradOutTensor.to(at::kFloat);
    }

    EXEC_NPU_CMD(aclnnVoxelPoolingTrainGrad, gradOutTensor, posMemo, batchSize, numPoints, numChannels, h, w, out);

    if (origin_dtype == at::kHalf) {
        out = out.to(at::kHalf);
    }
    return out;
}
