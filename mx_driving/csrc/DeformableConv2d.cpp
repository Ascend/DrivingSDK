// Copyright (c) 2024 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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

std::tuple<at::Tensor, at::Tensor> deformable_conv2d(const at::Tensor& input, const at::Tensor& offset,
    const at::Tensor& weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, int64_t groups, int64_t deformable_groups)
{
    TORCH_CHECK_NPU(input);
    TORCH_CHECK_NPU(offset);
    TORCH_CHECK_NPU(weight);
    TORCH_CHECK(input.dim() == 4, "input must to be a 4D Tensor, but got: ", input.dim());
    TORCH_CHECK(offset.dim() == 4, "offset has to be a 4D Tensor, but got: ", offset.dim());
    TORCH_CHECK(weight.dim() == 4, "weight has to be a 4D Tensor, but got: ", offset.dim());
    TORCH_CHECK(stride[0] > 0 && stride[1] > 0, "stride must be greater than 0");
    TORCH_CHECK(kernel_size[0] > 0 && kernel_size[1] > 0, "kernel_size must be greater than 0");
    TORCH_CHECK(dilation[0] > 0 && dilation[1] > 0, "dilation must be greater than 0");

    const at::Tensor& bias = at::Tensor();
    const at::Tensor& mask = at::Tensor();

    uint32_t n = input.size(0);
    uint32_t c_in = input.size(3);
    uint32_t h_in = input.size(1);
    uint32_t w_in = input.size(2);
    uint32_t h_out = offset.size(1);
    uint32_t w_out = offset.size(2);
    uint32_t c_out = weight.size(0);
    uint32_t kh = weight.size(1);
    uint32_t kw = weight.size(2);
    TORCH_CHECK(kh == kernel_size[0] && kw == kernel_size[1], "kernel size mismatch");
    TORCH_CHECK(groups > 0, "groups must be greater than 0");
    TORCH_CHECK(c_out % groups == 0, "weight's out channel should be divided by groups");
    TORCH_CHECK(c_in % groups == 0, "input's channel should be divided by groups");

    bool modulated = false;
    bool with_bias = false;

    at::Tensor output = at::empty({n, h_out, c_out, w_out}, input.options());
    at::Tensor offset_output = at::empty({n, h_out * w_out, groups, kh * kw, c_in / groups}, input.options());

    EXEC_NPU_CMD(aclnnDeformableConv2d, input, weight, bias, offset, mask, kernel_size, stride, padding, dilation,
        groups, deformable_groups, modulated, with_bias, output, offset_output);

    output = output.permute({0, 2, 1, 3});
    return std::tie(output, offset_output);
}
