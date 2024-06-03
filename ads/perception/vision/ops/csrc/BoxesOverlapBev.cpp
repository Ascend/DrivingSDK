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
#include "functions.h"

namespace {
constexpr int64_t N_IDX = 0;

void check_npu(const at::Tensor &boxes_a, const at::Tensor &boxes_b)
{
    TORCH_CHECK_NPU(boxes_a);
    TORCH_CHECK_NPU(boxes_b);
}
} // namespace

/**
 * @brief calculate overlap area of boxes
 * @param boxes_a: input boxes, 2D tensor(N, 5)
 * @param boxes_b: input boxes, 2D tensor(N, 5)
 * @return area_overlap: overlap area of boxes
 */
at::Tensor npu_boxes_overlap_bev(const at::Tensor &boxes_a, const at::Tensor &boxes_b)
{
    TORCH_CHECK(boxes_a.size(1) == 5, "boxes_a must be 2D tensor (N, 5)");
    TORCH_CHECK(boxes_b.size(1) == 5, "boxes_b must be 2D tensor (N, 5)");
    check_npu(boxes_a, boxes_b);

    auto boxes_a_num = boxes_a.size(N_IDX);
    auto boxes_b_num = boxes_b.size(N_IDX);
    auto output_size = {boxes_a_num, boxes_b_num};
    auto trans = true;
    auto is_clockwise = true;
    at::Tensor area_overlap = at::zeros(output_size, boxes_a.options());
    EXEC_NPU_CMD(aclnnBoxesOverlapBev, boxes_a, boxes_b, trans, is_clockwise, area_overlap);
    return area_overlap;
}
