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

std::tuple<at::Tensor, at::Tensor> multi_to_sparse(const at::Tensor& out_features,
    const at::Tensor& unique_indices_offset, const at::Tensor& sorted_idx_to_former_indices,
    const at::Tensor& outidx_pair)
{
    TORCH_CHECK_NPU(out_features);
    TORCH_CHECK_NPU(unique_indices_offset);
    TORCH_CHECK_NPU(sorted_idx_to_former_indices);
    TORCH_CHECK_NPU(outidx_pair);

    auto indices_size = unique_indices_offset.sizes();
    auto features_size = out_features.sizes();
    TORCH_CHECK(indices_size[0] > 1,
        "indices_size zeros dim must be greater than 1 expected but got indices_size[0] value: ", indices_size[0]);

    c10::SmallVector<int64_t, 2> out_size = {indices_size[0] - 1, features_size[1]};
    c10::SmallVector<int64_t, 2> out_idx_size = {indices_size[0] - 1, 8};
    at::Tensor sparse_value = at::empty(out_size, out_features.options());
    at::Tensor sparse_indices = at::empty(out_idx_size, unique_indices_offset.options());

    EXEC_NPU_CMD(aclnnToSparse, unique_indices_offset, out_features, sorted_idx_to_former_indices, outidx_pair,
        sparse_value, sparse_indices);
    return std::tie(sparse_value, sparse_indices);
}

std::tuple<at::Tensor, at::Tensor> multi_to_sparse_v2(const at::Tensor& features, const at::Tensor& weight,
    const at::Tensor& unique_indices_offset, const at::Tensor& sorted_idx_to_former_indices,
    const at::Tensor& outidx_pair)
{
    TORCH_CHECK_NPU(features);
    TORCH_CHECK_NPU(weight);
    TORCH_CHECK_NPU(unique_indices_offset);
    TORCH_CHECK_NPU(sorted_idx_to_former_indices);
    TORCH_CHECK_NPU(outidx_pair);

    auto features_size = features.sizes();
    auto weight_size = weight.sizes();
    auto indices_size = unique_indices_offset.sizes();

    TORCH_CHECK(indices_size[0] > 1,
        "indices_size zeros dim must be greater than 1 expected but got indices_size[0] value: ", indices_size[0]);

    c10::SmallVector<int64_t, 2> out_size = {indices_size[0] - 1, weight_size[4]};
    c10::SmallVector<int64_t, 2> out_idx_size = {indices_size[0] - 1, 8};

    at::Tensor sparse_value = at::empty(out_size, features.options());
    at::Tensor sparse_indices = at::empty(out_idx_size, unique_indices_offset.options());

    EXEC_NPU_CMD(aclnnToSparseV3, features, weight, unique_indices_offset, sorted_idx_to_former_indices, outidx_pair,
        sparse_value, sparse_indices);
    return std::tie(sparse_value, sparse_indices);
}
