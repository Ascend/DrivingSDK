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

/**
 * @brief sort key-value pairs
 * @param keys_in: keys to be sorted
 * @param values_in: values of corresponding keys
 * @param dim: dim to sort
 * @param descending: true-descending, false-ascending
 * @return (keys_out, values_out): (sorted keys, corresponding values of sorted keys)
 */
std::tuple<at::Tensor, at::Tensor> npu_sort_pairs(
    const at::Tensor& keys_in, const at::Tensor& values_in, int64_t dim, bool descending)
{
    TORCH_CHECK_NPU(keys_in);
    TORCH_CHECK_NPU(values_in);
    bool stable = true;
    at::Tensor keys_out = at::zeros_like(keys_in, keys_in.options());
    at::Tensor values_out = at::zeros_like(values_in, values_in.options());
    at::Tensor indices = at::zeros_like(values_in, values_in.options().dtype(at::kLong));
    EXEC_NPU_CMD(aclnnSort, keys_in, stable, dim, descending, keys_out, indices);
    EXEC_NPU_CMD(aclnnGather, values_in, dim, indices, values_out);
    return std::tie(keys_out, values_out);
}
