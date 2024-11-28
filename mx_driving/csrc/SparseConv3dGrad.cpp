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

std::tuple<at::Tensor, at::Tensor> npu_sparse_conv3d_grad(const at::Tensor& indices_offset,
    const at::Tensor& former_sorted_indices, const at::Tensor& feature, const at::Tensor& weight,
    const at::Tensor& grad)
{
    TORCH_CHECK_NPU(indices_offset);
    TORCH_CHECK_NPU(former_sorted_indices);
    TORCH_CHECK_NPU(feature);
    TORCH_CHECK_NPU(weight);
    TORCH_CHECK_NPU(grad);

    auto feature_size = feature.sizes();
    auto weight_size = weight.sizes();
    auto indices_size = indices_offset.sizes();

    int64_t kernelsum = 1;
    for (int32_t i = 0; i < weight_size.size() - 2; i++) {
        kernelsum *= weight_size[i];
    }
    int64_t kernelIC = weight_size[3];
    int64_t kernelOC = weight_size[4];

    at::Tensor weight_trans = weight.transpose(-1, -2).contiguous();

    c10::SmallVector<int64_t, 2> feature_grad_size = {feature_size[0], kernelIC};
    at::Tensor feature_grad = at::zeros(feature_grad_size, feature.options());
    at::Tensor weight_grad = at::zeros(weight_size, feature.options());

    EXEC_NPU_CMD(aclnnSparseConv3dGradV2, indices_offset, former_sorted_indices, feature, weight_trans, grad,
        feature_grad, weight_grad);
    return std::tie(feature_grad, weight_grad);
}
