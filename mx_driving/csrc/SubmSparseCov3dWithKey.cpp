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

at::Tensor npu_subm_sparse_conv3d_with_key(const at::Tensor& ouidx_offset, const at::Tensor& valid_indices,
                                           const at::Tensor& weight, const at::Tensor& feature, int indices_number,
                                           at::IntArrayRef kernel_size)
{
    auto weight_size = weight.sizes();
    int64_t kernelsum = 1;
    for (int32_t i = 0; i < weight_size.size() - 2; i++) {
        kernelsum *= weight_size[i];
    }
    c10::SmallVector<int64_t, 8> output_size = {indices_number, kernelsum, weight_size[4]};
    at::Tensor out = at::empty(output_size, weight.options()).fill_(0);
    int32_t inchannel = kernel_size[3];
    EXEC_NPU_CMD(aclnnSubmSparseConv3dWithKey, ouidx_offset, valid_indices, feature, kernel_size, inchannel, out);
    return out;
}
