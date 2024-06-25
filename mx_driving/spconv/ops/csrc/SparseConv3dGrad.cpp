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
#include "functions.h"
#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor> npu_sparse_conv3d_grad(const at::Tensor& indices_offset, const at::Tensor& former_sorted_indices,
                                                          const at::Tensor& feature, const at::Tensor& weight, const at::Tensor& grad)
{
    TORCH_CHECK_NPU(indices_offset);
    TORCH_CHECK_NPU(former_sorted_indices);
    TORCH_CHECK_NPU(feature);
    TORCH_CHECK_NPU(weight);
    TORCH_CHECK_NPU(grad);

    at::Tensor feature_grad = at::zeros(feature.sizes(), feature.options());
    at::Tensor weight_grad = at::zeros(weight.sizes(), feature.options());
    EXEC_NPU_CMD(aclnnSparseConv3dGrad, indices_offset, former_sorted_indices, feature, weight, grad, feature_grad, weight_grad);
    return std::tie(weight_grad, feature_grad);
}