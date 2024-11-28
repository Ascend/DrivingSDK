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

std::tuple<at::Tensor, at::Tensor> nms3d(const at::Tensor& boxes, double threshold)
{
    int32_t box_num = boxes.size(0);
    int32_t data_align = 16;
    int32_t mask_num = ((box_num - 1) / data_align + 1) * data_align;
    at::Tensor mask = at::empty({box_num, mask_num}, boxes.options().dtype(at::kShort));
    EXEC_NPU_CMD(aclnnNms3d, boxes, threshold, mask);

    at::Tensor keep = at::zeros({box_num}, mask.options());
    at::Tensor num_out = at::zeros(1, mask.options());
    EXEC_NPU_CMD(aclnnGatherNms3dMask, mask, keep, num_out);
    return std::tie(keep, num_out);
}
