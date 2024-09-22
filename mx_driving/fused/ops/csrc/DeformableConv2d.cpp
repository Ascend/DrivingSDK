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
#include "functions.h"

std::tuple<at::Tensor, at::Tensor> npu_deformable_conv2d(const at::Tensor &input, const at::Tensor &offset, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, int64_t groups, int64_t deformable_groups, bool modulated, bool xoffsets_transpose)
{
    TORCH_CHECK_NPU(input);
    TORCH_CHECK_NPU(offset);
    TORCH_CHECK_NPU(weight);
    TORCH_CHECK(input.dim() == 4, "input must to be a 4D Tensor, but got: ", input.dim());
    TORCH_CHECK(offset.dim() == 4, "offset has to be a 4D Tensor, but got: ", offset.dim());
    TORCH_CHECK(weight.dim() == 4, "weight has to be a 4D Tensor, but got: ", offset.dim());

    const at::Tensor &bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    at::Tensor bias_fp32 = bias;

    uint32_t B = input.size(0);
    uint32_t Cin = input.size(1);
    uint32_t Hin = input.size(2);
    uint32_t Win = input.size(3);
    uint32_t Hout = offset.size(2);
    uint32_t Wout = offset.size(3);
    uint32_t Cout = weight.size(0);
    uint32_t kh = weight.size(2);
    uint32_t kw = weight.size(3);

    c10::SmallVector<int64_t, SIZE> deformable_offsets_output_shape = {B, Hout * Wout, kh * kw, Cin};
    c10::SmallVector<int64_t, SIZE> output_shape = {B, Hout, Wout, Cout};
    c10::SmallVector<int64_t, SIZE> s = {stride[1], stride[2]};
    c10::SmallVector<int64_t, SIZE> p = {padding[1], padding[2]};
    c10::SmallVector<int64_t, SIZE> d = {dilation[1], dilation[2]};
    at::IntArrayRef strides = at::IntArrayRef(s);
    at::IntArrayRef pads = at::IntArrayRef(p);
    at::IntArrayRef dilations = at::IntArrayRef(d);
    at::IntArrayRef offsets_shape = at::IntArrayRef(deformable_offsets_output_shape);
    at::Tensor nhwc_input = input.permute({0, 2, 3, 1}).contiguous();
    at::Tensor nhwc_offset = offset.permute({0, 2, 3, 1}).contiguous();
    at::Tensor nhwc_weight = weight.permute({2, 3, 1, 0}).contiguous();
    at::Tensor output = at::empty(output_shape, input.options());
    at::Tensor deformable_offsets_output = at::empty(deformable_offsets_output_shape, input.options());

    EXEC_NPU_CMD(aclnnDeformableConv2d, nhwc_input, nhwc_offset, nhwc_weight, bias_fp32,
        kernel_size, strides, pads, dilations, groups, deformable_groups, modulated, deformable_offsets_output, output);

    output = output.permute({0, 3, 1, 2});

    if (xoffsets_transpose)
        deformable_offsets_output = deformable_offsets_output.view({B, Hout, Wout, kh, kw, Cin}).permute({0, 5, 1, 3, 2, 4}).contiguous().view({B, Cin, Hout * kh, Wout * kw});

    return std::tie(output, deformable_offsets_output);
}