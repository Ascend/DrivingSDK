// Copyright (c) 2023-2024 Huawei Technologies Co., Ltd
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

void roi_align_rotated_v2_forward_npu(const at::Tensor& input, const at::Tensor& rois_map, at::Tensor& output,
    double spatial_scale, int32_t sampling_ratio, int32_t pooled_height, int32_t pooled_width, bool aligned,
    bool clockwise)
{
    at::Tensor feature_map = input.permute({0, 2, 3, 1}).contiguous();
    at::Tensor rois = rois_map.permute({1, 0}).contiguous();
    EXEC_NPU_CMD(aclnnRoiAlignRotatedV2, feature_map, rois, spatial_scale, sampling_ratio, pooled_height, pooled_width,
        aligned, clockwise, output);
}
