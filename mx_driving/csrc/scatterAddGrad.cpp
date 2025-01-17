// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

at::Tensor npu_scatter_add_grad(at::Tensor& grad_out, at::Tensor& index, int32_t dim)
{
    TORCH_CHECK_NPU(grad_out);
    TORCH_CHECK_NPU(index);
    // construct the output tensor of the NPU
    auto index_size = index.sizes();
    auto grad_out_size = grad_out.sizes();
    auto index_dims = index.sizes().size();
    auto grad_out_dims = grad_out_size.size();
    TORCH_CHECK(grad_out.scalar_type() == at::kFloat,
        "grad_out: float32 tensor expected but got a tensor with dtype: ", grad_out.scalar_type());
    TORCH_CHECK(index.scalar_type() == at::kInt,
        "index: int32 tensor expected but got a tensor with dtype: ", index.scalar_type());
    TORCH_CHECK(grad_out_dims != 0 && index_dims != 0, "grad_out and index should not be empty");

    c10::SmallVector<int64_t, 8> grad_in_size;
    for (uint32_t i = 0; i < grad_out_dims; i++) {
        grad_in_size.push_back(grad_out_size[i]);
    }
    dim = (dim + index_dims) % index_dims;
    grad_in_size[dim] = index_size[dim];
    for (uint32_t i = 0; i < grad_out_dims; i++) {
        TORCH_CHECK(i >= index_dims || grad_in_size[i] == index_size[i], "the shape except dim should be the same");
    }
    uint64_t tail = 1;
    for (uint32_t i = index_dims; i < grad_out_dims; i++) {
        tail *= grad_out_size[i];
    }
    at::Tensor result;

    auto inputDim = index_dims - 1;
    grad_out = grad_out.transpose(dim, inputDim).contiguous();
    index = index.transpose(dim, inputDim).contiguous();

    auto grad_in_size_trans = grad_in_size;
    grad_in_size_trans[inputDim] = grad_in_size[dim];
    grad_in_size_trans[dim] = grad_in_size[inputDim];
    result = at::zeros(grad_in_size_trans, grad_out.options());
    EXEC_NPU_CMD(aclnnScatterAddGradV2, grad_out, index, inputDim, result);
    result = result.transpose(dim, inputDim).contiguous();
    return result;
}
