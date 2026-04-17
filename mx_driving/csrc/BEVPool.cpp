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
// bev_pool (v1) constants
constexpr int64_t N_IDX = 0;
constexpr int64_t C_IDX = 1;
constexpr int64_t C_IDX_BWD = 4;
constexpr int64_t N_INTERVAL_IDX = 0;

// bev_pool_v2 constants
constexpr int64_t C_IDX_V2 = 4;

// bev_pool_v3 constants
constexpr int64_t C_IDX_WITH_DEPTH = 4;
constexpr int64_t MINI_CHANNEL = 8;

// check_npu for bev_pool (v1)
void check_npu(const at::Tensor& feat, const at::Tensor& geom_feat, const at::Tensor& interval_lengths,
    const at::Tensor& interval_starts)
{
    TORCH_CHECK_NPU(feat);
    TORCH_CHECK_NPU(geom_feat);
    TORCH_CHECK_NPU(interval_lengths);
    TORCH_CHECK_NPU(interval_starts);
}

// check_npu for bev_pool_v2
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
 * @brief pillar pooling, bev_pool
 * @param feat: input feature, 2D tensor(n, c)
 * @param geom_feat: input coords, 2D tensor(n, 4)
 * @param interval_lengths: the number of points in each interval, 1D tensor(n_interval)
 * @param interval_starts: starting position for pooled point, 1D tensor(n_interval)
 * @param b: batch_size, int64
 * @param d: depth, int64
 * @param h: height, int64
 * @param w: width, int64
 * @return out: output feature, 5D tensor(b, d, h, w, c)
 */
at::Tensor npu_bev_pool(const at::Tensor& feat, const at::Tensor& geom_feat, const at::Tensor& interval_lengths,
    const at::Tensor& interval_starts, int64_t b, int64_t d, int64_t h, int64_t w)
{
    TORCH_CHECK(feat.dim() == 2, "feat must be 2D tensor(n, c)");
    TORCH_CHECK(geom_feat.dim() == 2, "coords must be 2D tensor(n, 4)");
    check_npu(feat, geom_feat, interval_lengths, interval_starts);

    auto n = geom_feat.size(N_IDX);
    auto c = feat.size(C_IDX);
    auto n_interval = interval_lengths.size(N_INTERVAL_IDX);
    TORCH_CHECK(
        interval_starts.size(N_INTERVAL_IDX) == n_interval, "interval_starts and interval_lengths must have same size");

    auto out = at::zeros({b, d, h, w, c}, feat.options());
    EXEC_NPU_CMD(aclnnBEVPool, feat, geom_feat, interval_lengths, interval_starts, b, d, h, w, c, out);
    return out;
}

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
    auto c = grad_out.size(C_IDX_BWD);
    auto n_interval = interval_lengths.size(N_INTERVAL_IDX);
    TORCH_CHECK(
        interval_starts.size(N_INTERVAL_IDX) == n_interval, "interval_starts and interval_lengths must have same size");

    auto grad_feat = at::zeros({n, c}, grad_out.options());
    EXEC_NPU_CMD(aclnnBEVPoolGrad, grad_out, geom_feat, interval_lengths, interval_starts, b, d, h, w, c, grad_feat);
    return grad_feat;
}

/**
 * @brief pillar pooling, bev_pool_v2
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
 * @return out: output feature, 5D tensor(b, d, h, w, c)
 */
at::Tensor npu_bev_pool_v2(const at::Tensor& depth, const at::Tensor& feat, const at::Tensor& ranks_depth,
    const at::Tensor& ranks_feat, const at::Tensor& ranks_bev, const at::Tensor& interval_lengths,
    const at::Tensor& interval_starts, int64_t b, int64_t d, int64_t h, int64_t w)
{
    check_npu(depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_lengths, interval_starts);
    auto c = feat.size(C_IDX_V2);
    auto out = at::zeros({b, d, h, w, c}, feat.options());
    EXEC_NPU_CMD(aclnnBEVPoolV2, depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_lengths, interval_starts, b,
        d, h, w, c, out);
    return out;
}

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
    auto c = feat.size(C_IDX_V2);

    EXEC_NPU_CMD(aclnnBEVPoolV2Grad, grad_out, depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_lengths,
        interval_starts, b, d, h, w, c, grad_depth, grad_feat);
    return std::make_tuple(grad_depth, grad_feat);
}

at::Tensor npu_bev_pool_v3(const c10::optional<at::Tensor>& depth, const at::Tensor& feat,
    const c10::optional<at::Tensor>& ranks_depth, const c10::optional<at::Tensor>& ranks_feat,
    const at::Tensor& ranks_bev, int64_t b, int64_t d, int64_t h, int64_t w)
{
    TORCH_CHECK_NPU(feat);
    TORCH_CHECK_NPU(ranks_bev);
    bool with_depth = depth.has_value();
    if (with_depth) {
        TORCH_CHECK_NPU(depth.value());
        TORCH_CHECK_NPU(ranks_depth.value());
        TORCH_CHECK_NPU(ranks_feat.value());
    }
    auto c = feat.size(with_depth ? C_IDX_WITH_DEPTH : C_IDX);
    TORCH_CHECK(c % MINI_CHANNEL == 0, "The channel of feature must be multiple of 8.");
    auto out = at::zeros({b, d, h, w, c}, feat.options());
    EXEC_NPU_CMD(aclnnBEVPoolV3, depth, feat, ranks_depth, ranks_feat, ranks_bev, with_depth, b, d, h, w, c, out);
    return out;
}

std::tuple<c10::optional<at::Tensor>, at::Tensor> npu_bev_pool_v3_backward(const at::Tensor& grad_out,
    const c10::optional<at::Tensor>& depth, const at::Tensor& feat, const c10::optional<at::Tensor>& ranks_depth,
    const c10::optional<at::Tensor>& ranks_feat, const at::Tensor& ranks_bev)
{
    TORCH_CHECK_NPU(grad_out);
    TORCH_CHECK_NPU(feat);
    TORCH_CHECK_NPU(ranks_bev);
    c10::optional<at::Tensor> grad_depth;
    bool with_depth = depth.has_value();
    if (with_depth) {
        TORCH_CHECK_NPU(depth.value());
        TORCH_CHECK_NPU(ranks_depth.value());
        TORCH_CHECK_NPU(ranks_feat.value());
        grad_depth = at::zeros_like(depth.value());
    }
    auto grad_feat = at::zeros_like(feat);
    EXEC_NPU_CMD(aclnnBEVPoolV3Grad, grad_out, depth, feat, ranks_depth, ranks_feat, ranks_bev, with_depth, grad_depth,
        grad_feat);
    return std::make_tuple(grad_depth, grad_feat);
}
