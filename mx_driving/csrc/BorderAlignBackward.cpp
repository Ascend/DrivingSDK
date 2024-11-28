// Copyright (c) OpenMMLab. All rights reserved.
// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

at::Tensor border_align_backward(const at::Tensor& grad_out, const at::Tensor& boxes, const at::Tensor& argmax_idx,
    int32_t pool_size, int32_t height, int32_t width)
{
    TORCH_CHECK_NPU(grad_out);
    TORCH_CHECK_NPU(boxes);
    TORCH_CHECK_NPU(argmax_idx);
    TORCH_CHECK(grad_out.dim() == 4, "grad_out.dim() must be 4, but got: ", grad_out.dim());
    TORCH_CHECK(boxes.dim() == 3, "idx.dim() must be 3, but got: ", boxes.dim());
    TORCH_CHECK(argmax_idx.dim() == 4, "argmax_idx.dim() must be 4, but got: ", argmax_idx.dim());

    int32_t batch_size = grad_out.size(0);
    int32_t feat_channels = grad_out.size(1) * 4;
    int32_t channels = grad_out.size(1);
    int32_t box_size = boxes.size(1);

    at::Tensor grad_input = at::zeros({batch_size, feat_channels, height, width}, grad_out.options());

    EXEC_NPU_CMD(aclnnBorderAlignGrad, grad_out, boxes, argmax_idx, channels, box_size, height, width, pool_size,
        batch_size, grad_input);
    return grad_input;
}
