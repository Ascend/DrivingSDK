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

    uint32_t n = static_cast<uint32_t>(input.size(0));
    uint32_t c_in = static_cast<uint32_t>(input.size(3));
    uint32_t h_in = static_cast<uint32_t>(input.size(1));
    uint32_t w_in = static_cast<uint32_t>(input.size(2));
    uint32_t h_out = static_cast<uint32_t>(offset.size(1));
    uint32_t w_out = static_cast<uint32_t>(offset.size(2));
    uint32_t c_out = static_cast<uint32_t>(weight.size(0));
    uint32_t kh = static_cast<uint32_t>(weight.size(1));
    uint32_t kw = static_cast<uint32_t>(weight.size(2));
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> deformable_conv2d_backward(const at::Tensor& input,
    const at::Tensor& weight, const at::Tensor& offset, const at::Tensor& offset_output, const at::Tensor& grad_y,
    at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, int64_t deformable_groups)
{
    TORCH_CHECK_NPU(input);
    TORCH_CHECK_NPU(offset);
    TORCH_CHECK_NPU(weight);
    TORCH_CHECK(input.dim() == 4, "input must to be a 4D Tensor, but got: ", input.dim());
    TORCH_CHECK(offset.dim() == 4, "offset has to be a 4D Tensor, but got: ", offset.dim());
    TORCH_CHECK(weight.dim() == 4, "weight has to be a 4D Tensor, but got: ", weight.dim());
    TORCH_CHECK(groups > 0, "groups must be greater than 0");

    const at::Tensor& bias = at::Tensor();
    const at::Tensor& grad_bias = at::Tensor();
    at::Tensor mask = at::Tensor();
    at::Tensor grad_mask = at::Tensor();

    auto input_sizes = input.sizes();   // n, h_in, w_in, c_in
    auto offset_sizes = offset.sizes(); // n, h_in, w_in, 2 * 9
    auto weight_sizes = weight.sizes(); // c_out, 9, c_in
    auto mask_sizes = mask.sizes();     // c_out, 9, c_in
    at::Tensor grad_input = at::zeros(input_sizes, input.options());
    at::Tensor grad_offset = at::empty(offset_sizes, offset.options());
    at::Tensor grad_weight = at::zeros(weight_sizes, weight.options());

    bool modulated = false;
    bool with_bias = false;

    EXEC_NPU_CMD(aclnnDeformableConv2dGrad, input, weight, bias, offset, mask, offset_output, grad_y, kernel_size,
        stride, padding, dilation, groups, deformable_groups, modulated, with_bias, grad_input, grad_weight, grad_bias,
        grad_offset, grad_mask);
    grad_input = grad_input.permute({0, 3, 1, 2});
    grad_weight = grad_weight.permute({0, 3, 1, 2});
    grad_offset = grad_offset.permute({0, 3, 1, 2});

    return std::tie(grad_input, grad_weight, grad_offset);
}
