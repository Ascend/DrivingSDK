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

std::tuple<at::Tensor, at::Tensor> knn(const at::Tensor& xyz, const at::Tensor& center_xyz, int32_t k, bool is_from_knn)
{
    TORCH_CHECK_NPU(xyz);
    TORCH_CHECK_NPU(center_xyz);
    TORCH_CHECK(center_xyz.dim() == 3, "center_xyz.dim() must be 3, but got: ", center_xyz.dim());

    at::Tensor dist = at::zeros({center_xyz.sizes()[0], center_xyz.sizes()[1], k}, center_xyz.options());
    at::Tensor idx = at::zeros({center_xyz.sizes()[0], center_xyz.sizes()[1], k}, center_xyz.options().dtype(at::kInt));
    EXEC_NPU_CMD_SYNC(aclnnKnn, xyz, center_xyz, is_from_knn, k, dist, idx);

    return std::tie(dist, idx);
}
