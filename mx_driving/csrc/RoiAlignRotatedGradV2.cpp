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

at::Tensor npu_roi_align_rotated_grad_v2(const at::Tensor& input, const at::Tensor& rois, const at::Tensor& grad_output,
    int32_t pooled_height, int32_t pooled_width, double spatial_scale, int32_t sampling_ratio, bool aligned,
    bool clockwise)
{
    auto ori_dtype = input.scalar_type();

    c10::SmallVector<int64_t, SIZE> grad_input_size = {input.size(0), input.size(2), input.size(3), input.size(1)};

    at::Tensor grad_input = at::zeros(grad_input_size, input.options());

    EXEC_NPU_CMD(aclnnRoiAlignRotatedGradV2, input, rois, grad_output, pooled_height, pooled_width, spatial_scale,
        sampling_ratio, aligned, clockwise, grad_input);

    return grad_input.to(ori_dtype);
}
