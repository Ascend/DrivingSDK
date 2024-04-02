// Copyright (c) 2024, Huawei Technologies.All rights reserved.
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

#ifndef PERCEPTION_FUSED_OPS_CSRC_FUNCTIONS_H_
#define PERCEPTION_FUSED_OPS_CSRC_FUNCTIONS_H_
#include <ATen/Tensor.h>
#include <torch/library.h>


at::Tensor npu_bev_pool(const at::Tensor& feat, const at::Tensor& geom_feat, const at::Tensor& interval_lengths,
    const at::Tensor& interval_starts, int64_t b, int64_t d, int64_t h, int64_t w);
at::Tensor npu_bev_pool_backward(const at::Tensor& grad_out, const at::Tensor& geom_feat,
    const at::Tensor& interval_lengths, const at::Tensor& interval_starts, int64_t b, int64_t d, int64_t h, int64_t w);

at::Tensor npu_bev_pool_v2(const at::Tensor& depth, const at::Tensor& feat, const at::Tensor& ranks_depth,
    const at::Tensor& ranks_feat, const at::Tensor& ranks_bev, const at::Tensor& interval_lengths,
    const at::Tensor& interval_starts, int64_t b, int64_t d, int64_t h, int64_t w);
std::tuple<at::Tensor, at::Tensor> npu_bev_pool_v2_backward(const at::Tensor& grad_out, const at::Tensor& depth,
    const at::Tensor& feat, const at::Tensor& ranks_depth, const at::Tensor& ranks_feat, const at::Tensor& ranks_bev,
    const at::Tensor& interval_lengths, const at::Tensor& interval_starts, int64_t b, int64_t d, int64_t h, int64_t w);
#endif // PERCEPTION_FUSED_OPS_CSRC_FUNCTIONS_H_
