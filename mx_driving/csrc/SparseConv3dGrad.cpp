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
constexpr int64_t MINI_CHANNELS = 16;
}

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
    int32_t unsumSize = 2;
    for (int32_t i = 0; i < static_cast<int32_t>(weight_size.size()) - unsumSize; i++) {
        kernelsum *= weight_size[i];
    }
    int64_t kernelIC = weight_size[3];
    int64_t kernelOC = weight_size[4];

    at::Tensor weight_trans = weight.transpose(-1, -2).contiguous();

    c10::SmallVector<int64_t, SIZE> feature_grad_size = {feature_size[0], kernelIC};
    at::Tensor feature_grad = at::zeros(feature_grad_size, feature.options());
    at::Tensor weight_grad = at::zeros(weight_size, feature.options());

    EXEC_NPU_CMD(aclnnSparseConv3dGradV2, indices_offset, former_sorted_indices, feature, weight_trans, grad,
        feature_grad, weight_grad);
    return std::tie(feature_grad, weight_grad);
}

std::tuple<at::Tensor, at::Tensor> npu_sparse_conv3d_grad_v2(const at::Tensor& former_sorted_indices,
    const at::Tensor& indices_offset, const at::Tensor& feature, const at::Tensor& weight, const at::Tensor& grad)
{
    TORCH_CHECK_NPU(former_sorted_indices);
    TORCH_CHECK_NPU(indices_offset);
    TORCH_CHECK_NPU(feature);
    TORCH_CHECK_NPU(weight);
    TORCH_CHECK_NPU(grad);

    TORCH_CHECK(indices_offset.size(0) > 1,
        "The length of 'indices_offset' must be greater than 1. Got: indices_offset length = ", indices_offset.size(0));

    TORCH_CHECK(former_sorted_indices.scalar_type() == at::kInt && indices_offset.scalar_type() == at::kInt,
        "'former_sorted_indices' and 'indices_offset' must be 'int32', but got: ", former_sorted_indices.scalar_type(),
        ", ", indices_offset.scalar_type());

    TORCH_CHECK(feature.scalar_type() == grad.scalar_type() && weight.scalar_type() == grad.scalar_type(),
        "Input data types must match, but got: feature type = ", feature.scalar_type(),
        ", weight type = ", weight.scalar_type(), ", grad type = ", grad.scalar_type());

    TORCH_CHECK(grad.size(0) + 1 == indices_offset.size(0),
        "The length of 'indices_offset' must match the number of output points. Got: indices_offset length = ",
        indices_offset.size(0), ", grad length = ", grad.size(0));

    TORCH_CHECK(feature.size(1) % MINI_CHANNELS == 0 && grad.size(1) % MINI_CHANNELS == 0,
        "Channels must be aligned to 16. Got: feature channels = ", feature.size(1),
        ", grad channels = ", grad.size(1));

    auto feature_size = feature.sizes();
    auto weight_size = weight.sizes();

    int64_t kernelIC = weight_size[3];
    int64_t kernelOC = weight_size[4];

    c10::SmallVector<int64_t, SIZE> feature_grad_size = {feature_size[0], kernelIC};
    auto dtype = feature.dtype();
    if (dtype == at::kHalf) {
        auto feature_fp32 = feature.to(at::kFloat);
        auto weight_fp32 = weight.to(at::kFloat);
        auto grad_fp32 = grad.to(at::kFloat);

        at::Tensor feature_grad_fp32 = at::zeros(feature_grad_size, feature.options().dtype(at::kFloat));
        at::Tensor weight_grad_fp32 = at::zeros(weight_size, feature.options().dtype(at::kFloat));
        EXEC_NPU_CMD(aclnnSparseConv3dGrad, feature_fp32, weight_fp32, grad_fp32, former_sorted_indices, indices_offset,
            feature_grad_fp32, weight_grad_fp32);

        at::Tensor feature_grad = feature_grad_fp32.to(at::kHalf);
        at::Tensor weight_grad = weight_grad_fp32.to(at::kHalf);

        return std::tie(feature_grad, weight_grad);
    } else {
        at::Tensor feature_grad = at::zeros(feature_grad_size, feature.options());
        at::Tensor weight_grad = at::zeros(weight_size, feature.options());

        EXEC_NPU_CMD(aclnnSparseConv3dGrad, feature, weight, grad, former_sorted_indices, indices_offset, feature_grad,
            weight_grad);

        return std::tie(feature_grad, weight_grad);
    }
}