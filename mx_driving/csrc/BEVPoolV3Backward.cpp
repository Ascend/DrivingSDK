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

namespace {
constexpr int64_t C_IDX = 4;

void check_npu(const at::Tensor& depth, const at::Tensor& feat, const at::Tensor& ranks_depth,
    const at::Tensor& ranks_feat, const at::Tensor& ranks_bev, const at::Tensor& grad_out)
{
    TORCH_CHECK_NPU(depth);
    TORCH_CHECK_NPU(feat);
    TORCH_CHECK_NPU(ranks_depth);
    TORCH_CHECK_NPU(ranks_feat);
    TORCH_CHECK_NPU(ranks_bev);
    TORCH_CHECK_NPU(grad_out);
}
} // namespace

std::tuple<at::Tensor, at::Tensor> npu_bev_pool_v3_backward(const at::Tensor& grad_out, const at::Tensor& depth,
    const at::Tensor& feat, const at::Tensor& ranks_depth, const at::Tensor& ranks_feat, const at::Tensor& ranks_bev)
{
    check_npu(depth, feat, ranks_depth, ranks_feat, ranks_bev, grad_out);
    auto depth_sizes = depth.sizes();
    auto feat_sizes = feat.sizes();
    auto grad_depth = at::zeros(depth_sizes, depth.options());
    auto grad_feat = at::zeros(feat_sizes, depth.options());

    EXEC_NPU_CMD(aclnnBEVPoolV3Grad, grad_out, depth, feat, ranks_depth, ranks_feat, ranks_bev, grad_depth, grad_feat);
    return std::make_tuple(grad_depth, grad_feat);
}
