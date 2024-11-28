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

void border_align(const at::Tensor& input, const at::Tensor& rois, at::Tensor& output, int32_t pooled_size)
{
    TORCH_CHECK(input.size(1) % 4 == 0, "The number of channels must be divisible by 4.");
    at::Tensor feature_map = input.permute({0, 2, 3, 1}).contiguous();
    at::Tensor rois_map = rois.contiguous();
    EXEC_NPU_CMD(aclnnBorderAlign, feature_map, rois_map, pooled_size, output);
}
