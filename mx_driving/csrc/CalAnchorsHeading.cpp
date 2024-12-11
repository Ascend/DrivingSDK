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

at::Tensor cal_anchors_heading(const at::Tensor& anchors, const at::Tensor& origin_pos)
{
    TORCH_CHECK_NPU(anchors);
    TORCH_CHECK_NPU(origin_pos);
    TORCH_CHECK(anchors.dim() == 4, "anchors must be a 4D Tensor, but got: ", anchors.dim());
    TORCH_CHECK(origin_pos.dim() == 2, "origin_pos must be a 2D Tensor, but got: ", origin_pos.dim());
    TORCH_CHECK(anchors.size(3) == 2, "the last dim of anchors must be 2, but got: ", anchors.size(3));

    uint32_t batch_size = anchors.size(0);
    uint32_t anchors_num = anchors.size(1);
    uint32_t seq_length = anchors.size(2);

    at::Tensor heading = at::empty({batch_size, anchors_num, seq_length}, anchors.options());

    EXEC_NPU_CMD(aclnnCalAnchorsHeading, anchors, origin_pos, heading);
    
    return heading;
}