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
constexpr int64_t C_IDX = 1;
constexpr int64_t C_IDX_WITH_DEPTH = 4;
} // namespace

at::Tensor npu_bev_pool_v3(const c10::optional<at::Tensor>& depth, const at::Tensor& feat,
    const c10::optional<at::Tensor>& ranks_depth, const c10::optional<at::Tensor>& ranks_feat,
    const at::Tensor& ranks_bev, int64_t b, int64_t d, int64_t h, int64_t w)
{
    TORCH_CHECK_NPU(feat);
    TORCH_CHECK_NPU(ranks_bev);
    bool with_depth = depth.has_value();
    auto c = feat.size(with_depth ? C_IDX_WITH_DEPTH : C_IDX);
    TORCH_CHECK(c % 8 == 0, "The channel of feature must be multiple of 8.");
    auto out = at::zeros({b, d, h, w, c}, feat.options());
    EXEC_NPU_CMD(aclnnBEVPoolV3, depth, feat, ranks_depth, ranks_feat, ranks_bev, with_depth, out);
    return out;
}
