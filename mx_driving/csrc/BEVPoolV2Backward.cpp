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
    const at::Tensor& ranks_feat, const at::Tensor& ranks_bev, const at::Tensor& interval_lengths,
    const at::Tensor& interval_starts)
{
    TORCH_CHECK_NPU(depth);
    TORCH_CHECK_NPU(feat);
    TORCH_CHECK_NPU(ranks_depth);
    TORCH_CHECK_NPU(ranks_feat);
    TORCH_CHECK_NPU(ranks_bev);
    TORCH_CHECK_NPU(interval_lengths);
    TORCH_CHECK_NPU(interval_starts);
}
} // namespace

/**
 * @brief pillar pooling, bev_pool_v2_backward
 * @param grad_out: input grad, 5D tensor(b, d, h, w, c)
 * @param depth: input depth, 5D tensor(b, n, d, h, w)
 * @param feat: input feature, 5D tensor(b, n, h, w, c)
 * @param ranks_depth: input depth rank, 1D tensor
 * @param ranks_feat: input feature rank, 1D tensor
 * @param ranks_bev: input bev rank, 1D tensor
 * @param interval_lengths: the number of points in each interval, 1D tensor(n_interval)
 * @param interval_starts: starting position for pooled point, 1D tensor(n_interval)
 * @param b: batch_size, int64
 * @param d: depth, int64
 * @param h: height, int64
 * @param w: width, int64
 * @return grad_depth: output grad, 5D tensor(b, n, d, h, w)
 * @return grad_feat: output grad, 5D tensor(b, n, h, w, c)
 */
std::tuple<at::Tensor, at::Tensor> npu_bev_pool_v2_backward(const at::Tensor& grad_out, const at::Tensor& depth,
    const at::Tensor& feat, const at::Tensor& ranks_depth, const at::Tensor& ranks_feat, const at::Tensor& ranks_bev,
    const at::Tensor& interval_lengths, const at::Tensor& interval_starts, int64_t b, int64_t d, int64_t h, int64_t w)
{
    check_npu(depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_lengths, interval_starts);
    auto depth_sizes = depth.sizes();
    auto feat_sizes = feat.sizes();
    auto grad_depth = at::zeros(depth_sizes, depth.options());
    auto grad_feat = at::zeros(feat_sizes, depth.options());
    auto c = feat.size(C_IDX);

    EXEC_NPU_CMD(aclnnBEVPoolV2Grad, grad_out, depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_lengths,
        interval_starts, b, d, h, w, c, grad_depth, grad_feat);
    return std::make_tuple(grad_depth, grad_feat);
}
