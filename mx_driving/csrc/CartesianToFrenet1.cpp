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

std::tuple<at::Tensor, at::Tensor> cartesian_to_frenet1(const at::Tensor& dist_vec)
{
    TORCH_CHECK_NPU(dist_vec);
    TORCH_CHECK(dist_vec.dim() == 4, "dist_vec.dim() must be 4, but got: ", dist_vec.dim());
    TORCH_CHECK(dist_vec.size(2) > 1, "Number of points in polyline must be greater than 1.");

    at::Tensor min_idx = at::zeros({dist_vec.size(0), dist_vec.size(1)}, dist_vec.options().dtype(at::kInt));
    at::Tensor back_idx = at::zeros({dist_vec.size(0), dist_vec.size(1)}, dist_vec.options().dtype(at::kInt));
    EXEC_NPU_CMD(aclnnCartesianToFrenet1, dist_vec, min_idx, back_idx);

    return std::tie(min_idx, back_idx);
}
