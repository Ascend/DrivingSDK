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
#ifndef COMMON_OPS_CSRC_FUNCTIONS_H_
#define COMMON_OPS_CSRC_FUNCTIONS_H_

#include <ATen/ATen.h>

std::tuple<at::Tensor, at::Tensor> knn(const at::Tensor& xyz, const at::Tensor& center_xyz, int32_t k, bool is_from_knn);

at::Tensor npu_three_interpolate(
    int b, int c, int m, int n, const at::Tensor& points, const at::Tensor& idx, const at::Tensor& weight);

at::Tensor npu_three_interpolate_backward(
    int b, int c, int n, int m, const at::Tensor& grad_out, const at::Tensor& idx, const at::Tensor& weight);

std::tuple<at::Tensor, at::Tensor> scatter_max_with_argmax_v2(
    const at::Tensor& updates, const at::Tensor& indices, c10::optional<at::Tensor> out);

at::Tensor npu_scatter_max_backward(const at::Tensor& x, const at::Tensor& segment_ids, const at::Tensor& num_segments);

at::Tensor npu_scatter(const at::Tensor& self, const at::Tensor& indices, const at::Tensor& updates, int64_t dim);

at::Tensor npu_scatter_mean_grad(const at::Tensor& grad_out, const at::Tensor& index, int32_t dim);

std::tuple<at::Tensor, at::Tensor> npu_scatter_mean(at::Tensor& src, at::Tensor& index,
                                                    c10::optional<at::Tensor> out, c10::optional<int> dim,
                                                    c10::optional<int> dim_size);
std::tuple<at::Tensor, at::Tensor> npu_sort_pairs(const at::Tensor &keys_in, const at::Tensor &values_in, int64_t dim, bool descending);

at::Tensor npu_hypot(const at::Tensor& input, const at::Tensor& other);

at::Tensor assign_score_withk(const at::Tensor& points, const at::Tensor& centers, const at::Tensor& scores, const at::Tensor& knn_idx,
                              int32_t aggregate, int32_t B, int32_t N, int32_t npoint, int32_t M, int32_t K, int32_t out_dim);

#endif // COMMON_OPS_CSRC_FUNCTIONS_H_
