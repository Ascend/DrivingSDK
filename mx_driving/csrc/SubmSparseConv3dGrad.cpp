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

at::Tensor npu_subm_sparse_conv3d_grad(const at::Tensor& ouidx_offset, const at::Tensor& valid_indices,
                                       const at::Tensor& weight, const at::Tensor& grad, int indices_number,
                                       at::IntArrayRef kernel_size)
{
    auto weight_size = weight.sizes();
    int64_t kernelsum = 1;
    int32_t unsumSize = 2;
    for (int32_t i = 0; i < static_cast<int32_t>(weight_size.size()) - unsumSize; i++) {
        kernelsum *= weight_size[i];
    }
    c10::SmallVector<int64_t, 8> output_size = {indices_number, kernelsum, weight_size[4]};
    at::Tensor out = at::empty(output_size, weight.options()).fill_(0);
    int32_t inchannel = kernel_size[3];
    EXEC_NPU_CMD(aclnnSubmSparseConv3dGrad, ouidx_offset, valid_indices, grad, kernel_size, inchannel, out);
    return out;
}


std::tuple<at::Tensor, at::Tensor> npu_subm_sparse_conv3d_grad_v2(
    const at::Tensor& features, 
    const at::Tensor& weight,
    const at::Tensor& grad_out_features,
    const at::Tensor& indices_offset
    )
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

        EXEC_NPU_CMD(aclnnSubmSparseConv3dGradV2, features, weight, grad_out_features, indices_offset, 
            features_grad, weight_grad);

    } else {
        at::Tensor weight_grad_fp32 = at::zeros(weight_size, weight.options().dtype(at::kFloat));

        EXEC_NPU_CMD(aclnnSubmSparseConv3dGradV2, features, weight, grad_out_features, indices_offset, 
            features_grad, weight_grad_fp32);

        weight_grad = weight_grad_fp32.to(at::kHalf);
    }
    
    return std::tie(features_grad, weight_grad);
}