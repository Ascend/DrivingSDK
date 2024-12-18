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

constexpr size_t C_LIMIT = 64;
constexpr size_t X_NUM_LIMIT = 1000000000;

at::Tensor npu_max_pool2d(const at::Tensor& x, int kernel_size, int stride, int padding)
{
    TORCH_CHECK_NPU(x);
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf,
        "x: float32 or float16 tensor expected but got a tensor with dtype: ", x.scalar_type());
    TORCH_CHECK(kernel_size == 3, "kernel_size: expected 3 but got: ", kernel_size);
    TORCH_CHECK(stride == 2, "stride: expected 2 but got: ", stride);
    TORCH_CHECK(padding == 1, "padding: expected 1 but got: ", padding);

    TORCH_CHECK(x.dim() == 4, "x_trans.dim() must be 4, but got: ", x.dim());
    auto x_size = x.sizes();
    auto batch = x_size[0];
    auto channel = x_size[1];
    auto height = x_size[2];
    auto width = x_size[3];

    auto output_height = (height + 1) / 2;
    auto output_width = (width + 1) / 2;
    if (channel < C_LIMIT || height == 1 || width == 1 || x.numel() > X_NUM_LIMIT) {
        auto output_size = {batch, channel, output_height, output_width};
        at::Tensor y = at::empty(output_size, x.options());

        int64_t mask_H = kernel_size * kernel_size;
        const int64_t BLOCKSIZE = 16;
        int64_t mask_W = (output_height * output_width + BLOCKSIZE - 1) / BLOCKSIZE + 1;

        c10::SmallVector<int64_t, SIZE> indices_size = {batch, channel, mask_H, mask_W * 32};
        at::Tensor indices = at::empty(indices_size, x.options().dtype(at::kChar));

        c10::SmallVector<int64_t, N> kernel_sizes = {kernel_size, kernel_size};
        c10::SmallVector<int64_t, N> stride_sizes = {stride, stride};
        c10::SmallVector<int64_t, N> padding_sizes = {padding, padding};
        c10::SmallVector<int64_t, N> dilation_sizes = {1, 1};
        at::IntArrayRef kernels = at::IntArrayRef(kernel_sizes);
        at::IntArrayRef strides = at::IntArrayRef(stride_sizes);
        at::IntArrayRef paddings = at::IntArrayRef(padding_sizes);
        at::IntArrayRef dilations = at::IntArrayRef(dilation_sizes);
        bool ceil_mode = false;

        EXEC_NPU_CMD(aclnnMaxPool2dWithMask, x, kernels, strides, paddings, dilations, ceil_mode, y, indices);
        return y;
    } else {
        if (x.scalar_type() == at::kFloat) {
            TORCH_CHECK(channel % 8 == 0, "channel: expected 8X when dtype is fp32  but got: ", channel);
        } else if (x.scalar_type() == at::kHalf) {
            TORCH_CHECK(channel % 16 == 0, "channel: expected 16X when dtype is fp16 but got: ", channel);
        }

        at::Tensor x_trans = x.permute({0, 2, 3, 1});

        auto output_size = {batch, output_height, output_width, channel};
        at::Tensor y_trans = at::empty(output_size, x.options());

        EXEC_NPU_CMD(aclnnMaxPool2d, x_trans, y_trans);
        at::Tensor y = y_trans.permute({0, 3, 1, 2});
        return y;
    }
}
