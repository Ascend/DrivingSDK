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
constexpr int64_t N_IDX = 0;
constexpr int64_t C_IDX = 4;
constexpr int64_t N_INTERVAL_IDX = 0;

void check_npu(const at::Tensor& grad_out, const at::Tensor& geom_feat, const at::Tensor& interval_lengths,
    const at::Tensor& interval_starts)
{
    TORCH_CHECK_NPU(grad_out);
    TORCH_CHECK_NPU(geom_feat);
    TORCH_CHECK_NPU(interval_lengths);
    TORCH_CHECK_NPU(interval_starts);
}
} // namespace

/**
 * @brief pillar pooling, bev_pool_backward
 * @param grad_out: input grad, 5D tensor(b, d, h, w, c)
 * @param geom_feat: input coords, 2D tensor(n, 4)
 * @param interval_lengths: the number of points in each interval, 1D tensor(n_interval)
 * @param interval_starts: starting position for pooled point, 1D tensor(n_interval)
 * @param b: batch_size, int64
 * @param d: depth, int64
 * @param h: height, int64
 * @param w: width, int64
 * @return grad_feat: output grad, 2D tensor(n, c)
 */
at::Tensor npu_bev_pool_backward(const at::Tensor& grad_out, const at::Tensor& geom_feat,
    const at::Tensor& interval_lengths, const at::Tensor& interval_starts, int64_t b, int64_t d, int64_t h, int64_t w)
{
    TORCH_CHECK(grad_out.dim() == 5, "grad_out must be 5D tensor(b, d, h, w, c)");
    TORCH_CHECK(geom_feat.dim() == 2, "coords must be 2D tensor(n, 4)");
    check_npu(grad_out, geom_feat, interval_lengths, interval_starts);
    auto n = geom_feat.size(N_IDX);
    auto c = grad_out.size(C_IDX);
    auto n_interval = interval_lengths.size(N_INTERVAL_IDX);
    TORCH_CHECK(
        interval_starts.size(N_INTERVAL_IDX) == n_interval, "interval_starts and interval_lengths must have same size");

    auto grad_feat = at::zeros({n, c}, grad_out.options());
    EXEC_NPU_CMD(aclnnBEVPoolGrad, grad_out, geom_feat, interval_lengths, interval_starts, b, d, h, w, c, grad_feat);
    return grad_feat;
}
