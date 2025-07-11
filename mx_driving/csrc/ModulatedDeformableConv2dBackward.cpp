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
constexpr int8_t INPUT_DIM = 4;
constexpr int16_t CHANNEL_256 = 256;
constexpr int16_t CHANNEL_512 = 512;
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> modulated_deformable_conv2d_backward(
    const at::Tensor& input, const at::Tensor& offset, const at::Tensor& mask, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt, const at::Tensor& offset_output, const at::Tensor& grad_y,
    at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, int64_t deformable_groups, int64_t with_bias)
{
    TORCH_CHECK_NPU(input);
    TORCH_CHECK_NPU(offset);
    TORCH_CHECK_NPU(weight);
    TORCH_CHECK(input.dim() == INPUT_DIM, "input must to be a 4D Tensor, but got: ", input.dim());
    TORCH_CHECK(offset.dim() == INPUT_DIM, "offset has to be a 4D Tensor, but got: ", offset.dim());
    TORCH_CHECK(mask.dim() == INPUT_DIM, "mask has to be a 4D Tensor, but got: ", mask.dim());
    TORCH_CHECK(weight.dim() == INPUT_DIM, "weight has to be a 4D Tensor, but got: ", weight.dim());
    TORCH_CHECK(stride[0] > 0 && stride[1] > 0, "stride must be greater than 0");
    TORCH_CHECK(kernel_size[0] > 0 && kernel_size[1] > 0, "kernel_size must be greater than 0");
    TORCH_CHECK(dilation[0] > 0 && dilation[1] > 0, "dilation must be greater than 0");

    const at::Tensor& bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    at::Tensor grad_bias = at::Tensor();

    auto input_sizes = input.sizes();   // n, h_in, w_in, c_in
    auto offset_sizes = offset.sizes(); // n, h_in, w_in, 2 * 9
    auto weight_sizes = weight.sizes(); // c_out, 9, c_in
    auto mask_sizes = mask.sizes();     // c_out, 9, c_in

    at::Tensor grad_input = at::zeros(input_sizes, input.options());
    at::Tensor grad_offset = at::empty({offset_sizes[0], offset_sizes[1], offset_sizes[2], 2, kernel_size[0], kernel_size[1]}, offset.options());
    at::Tensor grad_weight = at::zeros(weight_sizes, weight.options());
    at::Tensor grad_mask = at::zeros(mask_sizes, mask.options());

    bool modulated = true;
    uint32_t c_in = static_cast<uint32_t>(input.size(3));
    if ((groups == 1) && ((c_in == CHANNEL_256) || (c_in == CHANNEL_512))) {
        EXEC_NPU_CMD(aclnnDeformableConv2dGradV2, input, weight, bias, offset, mask, grad_y, kernel_size,
            stride, padding, dilation, groups, deformable_groups, modulated, with_bias, grad_input, grad_weight, grad_bias,
            grad_offset, grad_mask);
    } else {
        EXEC_NPU_CMD(aclnnDeformableConv2dGrad, input, weight, bias, offset, mask, offset_output, grad_y, kernel_size,
            stride, padding, dilation, groups, deformable_groups, modulated, with_bias, grad_input, grad_weight, grad_bias,
            grad_offset, grad_mask);
    }
    
    grad_input = grad_input.permute({0, 3, 1, 2});
    grad_weight = grad_weight.permute({0, 3, 1, 2});
    grad_offset = grad_offset.permute({0, 4, 5, 3, 1, 2});
    grad_mask = grad_mask.permute({0, 3, 1, 2});

    return std::tie(grad_input, grad_weight, grad_bias, grad_offset, grad_mask);
}