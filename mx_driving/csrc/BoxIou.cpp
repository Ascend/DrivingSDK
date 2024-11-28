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

namespace {
constexpr int64_t N_IDX = 0;

void check_npu(const at::Tensor& boxes_a, const at::Tensor& boxes_b)
{
    TORCH_CHECK_NPU(boxes_a);
    TORCH_CHECK_NPU(boxes_b);
}
} // namespace

/**
 * @brief calculate iou of boxes
 * @param boxes_a: input boxes, 2D tensor(N, 8), format: (x1, y1, ..., x4, y4)
 * @param boxes_b: input boxes, 2D tensor(N, 8), format: (x1, y1, ..., x4, y4)
 * @param mode_flag: 0-iou (intersection over union), 1-iof (intersection over foreground)
 * @param aligned: False-calculate between each box of boxes_a and boxes_b, True-calculate between each aligned pair of boxes_a and boxes_b
 * @return ious: iou of boxes
 */
at::Tensor npu_box_iou_quadri(
    const at::Tensor& boxes_a, const at::Tensor& boxes_b, const int64_t mode_flag, const bool aligned)
{
    TORCH_CHECK(boxes_a.size(1) == 8, "boxes_a must be 2D tensor (N, 8)");
    TORCH_CHECK(boxes_b.size(1) == 8, "boxes_b must be 2D tensor (N, 8)");
    check_npu(boxes_a, boxes_b);

    auto boxes_a_num = boxes_a.size(N_IDX);
    auto boxes_b_num = boxes_b.size(N_IDX);
    c10::SmallVector<int64_t, SIZE> output_size = {boxes_a_num};
    if (!aligned) {
        output_size.push_back(boxes_b_num);
    }
    at::Tensor ious = at::zeros(output_size, boxes_a.options());
    EXEC_NPU_CMD(aclnnBoxIou, boxes_a, boxes_b, mode_flag, aligned, ious);
    return ious;
}

/**
 * @brief calculate iou of boxes
 * @param boxes_a: input boxes, 2D tensor(N, 5), format: (x_center, y_center, width, height, angle)
 * @param boxes_b: input boxes, 2D tensor(N, 5), format: (x_center, y_center, width, height, angle)
 * @param mode_flag: 0-iou (intersection over union), 1-iof (intersection over foreground)
 * @param aligned: False-calculate between each box of boxes_a and boxes_b, True-calculate between each aligned pair of boxes_a and boxes_b
 * @return ious: iou of boxes
 */
at::Tensor npu_box_iou_rotated(
    const at::Tensor& boxes_a, const at::Tensor& boxes_b, const int64_t mode_flag, const bool aligned)
{
    TORCH_CHECK(boxes_a.size(1) == 5, "boxes_a must be 2D tensor (N, 5)");
    TORCH_CHECK(boxes_b.size(1) == 5, "boxes_b must be 2D tensor (N, 5)");
    check_npu(boxes_a, boxes_b);

    auto boxes_a_num = boxes_a.size(N_IDX);
    auto boxes_b_num = boxes_b.size(N_IDX);
    c10::SmallVector<int64_t, SIZE> output_size = {boxes_a_num};
    if (!aligned) {
        output_size.push_back(boxes_b_num);
    }
    at::Tensor ious = at::zeros(output_size, boxes_a.options());
    EXEC_NPU_CMD(aclnnBoxIou, boxes_a, boxes_b, mode_flag, aligned, ious);
    return ious;
}
