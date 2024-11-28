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

at::Tensor npu_points_in_box(const at::Tensor& boxes, const at::Tensor& pts)
{
    TORCH_CHECK(pts.size(0) == 1 && boxes.size(0) == 1, "points_in_box npu only support batch size = 1");
    TORCH_CHECK(boxes.size(1) <= 200, "boxes is larger than 200");
    c10::SmallVector<int64_t, 8> output_size = {pts.size(0), pts.size(1)};
    at::Tensor out = at::empty(output_size, pts.options().dtype(at::kInt));
    auto boxes_trans = boxes.transpose(1, 2).contiguous();
    EXEC_NPU_CMD(aclnnPointsInBox, boxes_trans, pts, out);
    return out;
}
