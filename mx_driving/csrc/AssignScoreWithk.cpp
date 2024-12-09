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

void assign_score_withk(const at::Tensor& points, const at::Tensor& centers, const at::Tensor& scores,
    const at::Tensor& knn_idx, at::Tensor& output, int32_t B, int32_t N, int32_t npoint, int32_t M, int32_t K,
    int32_t out_dim, int32_t aggregate)
{
    TORCH_CHECK_NPU(points);
    TORCH_CHECK_NPU(centers);
    TORCH_CHECK_NPU(scores);
    TORCH_CHECK_NPU(knn_idx);
    TORCH_CHECK_NPU(output);
    TORCH_CHECK(points.dim() == 4, "points.dim() must be 4, but got: ", points.dim());
    TORCH_CHECK(centers.dim() == 4, "centers.dim() must be 4, but got: ", centers.dim());
    TORCH_CHECK(scores.dim() == 4, "scores.dim() must be 4, but got: ", scores.dim());
    TORCH_CHECK(knn_idx.dim() == 3, "knn_idx.dim() must be 3, but got: ", knn_idx.dim());
    TORCH_CHECK(N >= npoint, "The number of whole points must be larger than or equal to the number of sample points.");
    TORCH_CHECK(N >= K, "The number of whole points must be larger than or equal to the number of neighbors.");
    TORCH_CHECK(aggregate == 0, "Only support 'sum' method, other aggregate methods are not supported now.");

    at::Tensor points_trans = points.permute({0, 3, 1, 2});
    at::Tensor centers_trans = centers.permute({0, 3, 1, 2});

    EXEC_NPU_CMD_SYNC(aclnnAssignScoreWithk, points_trans, centers_trans, scores, knn_idx, B, N, npoint, M, K, out_dim,
        aggregate, output);
}

void assign_score_withk_grad(
    const at::Tensor& grad_out,
    const at::Tensor& points,
    const at::Tensor& centers,
    const at::Tensor& scores,
    const at::Tensor& knn_idx,
    at::Tensor & grad_points,
    at::Tensor & grad_centers,
    at::Tensor & grad_scores,
    int32_t B,
    int32_t N,
    int32_t npoint,
    int32_t M,
    int32_t K,
    int32_t out_dim,
    int32_t aggregate
    )
{
    TORCH_CHECK_NPU(points);
    TORCH_CHECK_NPU(centers);
    TORCH_CHECK_NPU(scores);
    TORCH_CHECK_NPU(knn_idx);
    TORCH_CHECK_NPU(grad_out);
    TORCH_CHECK(points.dim() == 4, "points.dim() must be 4, but got: ", points.dim());
    TORCH_CHECK(centers.dim() == 4, "centers.dim() must be 4, but got: ", centers.dim());
    TORCH_CHECK(scores.dim() == 4, "scores.dim() must be 4, but got: ", scores.dim());
    TORCH_CHECK(knn_idx.dim() == 3, "knn_idx.dim() must be 3, but got: ", knn_idx.dim());
    TORCH_CHECK(grad_out.dim() == 4, "grad_out.dim() must be 4, but got: ", grad_out.dim());
    TORCH_CHECK(N >= npoint, "The number of whole points must be larger than or equal to the number of sample points.");
    TORCH_CHECK(N >= K, "The number of whole points must be larger than or equal to the number of neighbors.");
    TORCH_CHECK(M * out_dim <= 5000, "The size of M or K is too large.");
    TORCH_CHECK(aggregate == 0, "Only support 'sum' method, other aggregate methods are not supported now.");

    at::Tensor grad_out_trans = grad_out.permute({0, 2, 3, 1});

    EXEC_NPU_CMD(aclnnAssignScoreWithkGrad, grad_out_trans, points, centers, scores, knn_idx, B, N, npoint, M, K, out_dim, aggregate, grad_scores, grad_points, grad_centers);
}
