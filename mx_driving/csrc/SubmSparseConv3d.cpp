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
constexpr size_t TOTAL_CAPACITY = 8;
constexpr uint8_t KERNEL_SIZE_1 = 1;
constexpr uint8_t KERNEL_SIZE_3 = 3;
constexpr uint8_t KERNEL_SIZE_5 = 5;
constexpr uint32_t KERNEL_SIZE_IDX_0 = 0;
constexpr uint32_t KERNEL_SIZE_IDX_1 = 1;
constexpr uint32_t KERNEL_SIZE_IDX_2 = 2;
}; // namespace

std::tuple<at::Tensor, at::Tensor> npu_subm_sparse_conv3d_v3(const at::Tensor& feature, const at::Tensor& weight,
    const at::Tensor& indices, const at::Tensor& indices_offset, const at::Tensor& map1, const at::Tensor& map2,
    at::IntArrayRef kernel_size, int in_channels, int out_channels, at::IntArrayRef out_spatial_shape, int batch_size,
    int with_key)
{
    TORCH_CHECK_NPU(feature);
    TORCH_CHECK_NPU(indices);
    TORCH_CHECK_NPU(weight);

    auto indices_size = indices.sizes();
    int64_t kernelsum = 1;
    for (int32_t i = 0; i < static_cast<int32_t>(kernel_size.size()); i++) {
        kernelsum *= kernel_size[i];
    }
    int64_t outputsum = indices_size[0] * kernelsum;

    c10::SmallVector<int64_t, TOTAL_CAPACITY> output_size = {indices_size[0], out_channels};
    c10::SmallVector<int64_t, TOTAL_CAPACITY> indices_out_size = {outputsum};

    at::Tensor feature_out = at::zeros(output_size, feature.options());
    at::Tensor out_indices_offset = at::empty(indices_out_size, feature.options().dtype(at::kInt));

    EXEC_NPU_CMD(aclnnSubmSparseConv3dV3, feature, weight, indices, indices_offset, map1, map2, kernel_size,
        in_channels, out_channels, out_spatial_shape, batch_size, with_key, feature_out, out_indices_offset);

    return std::tie(feature_out, out_indices_offset);
}

std::tuple<at::Tensor, at::Tensor> npu_subm_sparse_conv3d_grad_v2(const at::Tensor& features, const at::Tensor& weight,
    const at::Tensor& grad_out_features, const at::Tensor& indices_offset)
{
    TORCH_CHECK_NPU(features);
    TORCH_CHECK_NPU(weight);
    TORCH_CHECK_NPU(grad_out_features);
    TORCH_CHECK_NPU(indices_offset);

    auto features_size = features.sizes();
    auto weight_size = weight.sizes();

    at::Tensor features_grad = at::zeros(features_size, features.options());
    at::Tensor weight_grad = at::zeros(weight_size, weight.options());

    // zero init
    if (features.options().dtype() == at::kFloat) {
        EXEC_NPU_CMD(aclnnSubmSparseConv3dGradV2, features, weight, grad_out_features, indices_offset, features_grad,
            weight_grad);
    } else {
        at::Tensor weight_grad_fp32 = at::zeros(weight_size, weight.options().dtype(at::kFloat));
        EXEC_NPU_CMD(aclnnSubmSparseConv3dGradV2, features, weight, grad_out_features, indices_offset, features_grad,
            weight_grad_fp32);

        weight_grad = weight_grad_fp32.to(at::kHalf);
    }

    return std::tie(features_grad, weight_grad);
}