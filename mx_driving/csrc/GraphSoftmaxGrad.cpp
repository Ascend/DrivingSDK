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
#include "csrc/utils.h"
#include "csrc/functions.h"

constexpr size_t EDGE_NUM_DIM = 0;
constexpr size_t SRC_FEATURE_DIM = 1;
constexpr size_t FEATURE_NUM = 8;

at::Tensor graph_softmax_grad(const at::Tensor& index,
    const at::Tensor& softmax_out, const at::Tensor& grad_output, int32_t node_num)
{
    TORCH_CHECK(index.scalar_type() == at::kInt,
        "index: int32 tensor expected but got a tensor with dtype: ", index.scalar_type());
    TORCH_CHECK(softmax_out.scalar_type() == at::kFloat,
        "softmax_out: float32 tensor expected but got a tensor with dtype: ", softmax_out.scalar_type());
    TORCH_CHECK(grad_output.scalar_type() == at::kFloat,
        "grad_output: float32 tensor expected but got a tensor with dtype: ", grad_output.scalar_type());

    auto softmax_out_size = softmax_out.sizes();
    auto index_size = index.sizes();
    auto edge_num = softmax_out_size[EDGE_NUM_DIM];
    auto feature_num = softmax_out_size[SRC_FEATURE_DIM];
    auto index_edge = index_size[EDGE_NUM_DIM];

    TORCH_CHECK(feature_num == FEATURE_NUM, "dim 2 of softmax_out tensor is invalid!");
    TORCH_CHECK(edge_num == index_edge, "softmax_out tensor and index tensor should have same Edge num.");

    at::Tensor grad_src = at::zeros({edge_num, feature_num}, softmax_out.options().dtype(at::kFloat));
    at::Tensor reduce_sum = at::zeros({node_num, feature_num}, softmax_out.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnGraphSoftmaxGrad, index, softmax_out, grad_output, reduce_sum, node_num, grad_src);

    return grad_src;
}
