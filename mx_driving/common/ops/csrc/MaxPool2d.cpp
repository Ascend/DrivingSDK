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

at::Tensor npu_max_pool2d(const at::Tensor& x_trans, int kernel_size, int stride, int padding)
{
    TORCH_CHECK_NPU(x_trans);
    TORCH_CHECK(x_trans.scalar_type() == at::kFloat,
        "x: float32 tensor expected but got a tensor with dtype: ", x_trans.scalar_type());
    TORCH_CHECK(kernel_size == 3, "kernel_size: expected 3 but got: ", kernel_size);
    TORCH_CHECK(stride == 2, "stride: expected 2 but got: ", stride);
    TORCH_CHECK(padding == 1, "padding: expected 1 but got: ", padding);

    TORCH_CHECK(x_trans.dim() == 4, "x_trans.dim() must be 4, but got: ", x_trans.dim());
    auto x_size = x_trans.sizes();
    auto batch = x_size[0];
    auto height = x_size[1];
    auto width = x_size[2];
    auto channel = x_size[3];
    TORCH_CHECK(channel % 8 == 0, "channel: expected 8X but got: ", channel);

    auto output_size = {batch, (height + 1) / 2, (width + 1) / 2, channel};
    at::Tensor y_trans = at::empty(output_size, x_trans.options());

    EXEC_NPU_CMD(aclnnMaxPool2d, x_trans, y_trans);
    return y_trans;
}
