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
#include "functions.h"

at::Tensor assign_score_withk(
    const at::Tensor& points,
    const at::Tensor& centers,
    const at::Tensor& scores,
    const at::Tensor& knn_idx,
    int32_t aggregate,
    int32_t B,
    int32_t N,
    int32_t npoint,
    int32_t M,
    int32_t K,
    int32_t out_dim
    )
{
    TORCH_CHECK_NPU(points);
    TORCH_CHECK_NPU(centers);
    TORCH_CHECK_NPU(scores);
    TORCH_CHECK_NPU(knn_idx);
    TORCH_CHECK(points.dim() == 5, "points.dim() must be 5, but got: ", points.dim());
    TORCH_CHECK(centers.dim() == 4, "centers.dim() must be 4, but got: ", centers.dim());
    TORCH_CHECK(scores.dim() == 5, "scores.dim() must be 5, but got: ", scores.dim());

    at::Tensor output = at::empty({npoint, B, K, out_dim}, points.options());
    EXEC_NPU_CMD_SYNC(aclnnAssignScoreWithk, points, centers, scores, knn_idx, aggregate, B, N, npoint, M, K, out_dim, output);

    return output;
}