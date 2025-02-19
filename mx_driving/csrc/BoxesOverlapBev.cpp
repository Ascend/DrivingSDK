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
constexpr int32_t BOXES_NUM_DIM = 0;
constexpr int32_t FORMAT_FLAG_XYXYR = 0;
constexpr int32_t FORMAT_FLAG_XYWHR = 1;
constexpr int32_t FORMAT_FLAG_XYZXYZR = 2;
constexpr int32_t FORMAT_FLAG_XYZWHDR = 3;
constexpr int32_t UNIT_FLAG_RADIAN = 0;
constexpr int32_t UNIT_FLAG_DEGREE = 1;
constexpr int32_t MODE_FLAG_OVERLAP = 0;
constexpr int32_t MODE_FLAG_IOU = 1;
constexpr float PI = 3.14159265358979323846;

void check_npu(const at::Tensor& boxes_a, const at::Tensor& boxes_b)
{
    TORCH_CHECK_NPU(boxes_a);
    TORCH_CHECK_NPU(boxes_b);
}
} // namespace

/**
 * @brief calculate area overlap/iou/iof of boxes
 * @param boxes_a: input boxes, 2D tensor (N, 5)/(N, 7)
 * @param boxes_b: input boxes, 2D tensor (N, 5)/(N, 7)
 * @return res: area overlap/iou/iof of boxes
 */
at::Tensor npu_boxes_overlap_bev(const at::Tensor& boxes_a, const at::Tensor& boxes_b,
                                 int32_t format_flag, int32_t unit_flag, bool clockwise,
                                 int32_t mode_flag, bool aligned, double margin)
{
    check_npu(boxes_a, boxes_b);

    if (format_flag == FORMAT_FLAG_XYXYR || format_flag == FORMAT_FLAG_XYWHR) {
        TORCH_CHECK(boxes_a.size(1) == 5, "boxes_a must be 2D tensor (N, 5)");
        TORCH_CHECK(boxes_b.size(1) == 5, "boxes_b must be 2D tensor (N, 5)");
    } else {
        TORCH_CHECK(boxes_a.size(1) == 7, "boxes_a must be 2D tensor (N, 7)");
        TORCH_CHECK(boxes_b.size(1) == 7, "boxes_b must be 2D tensor (N, 7)");
    }

    auto boxes_a_num = boxes_a.size(BOXES_NUM_DIM);
    auto boxes_b_num = boxes_b.size(BOXES_NUM_DIM);
    c10::SmallVector<int64_t, SIZE> output_size = {boxes_a_num};
    if (!aligned) {
        output_size.push_back(boxes_b_num);
    }
    at::Tensor res = at::zeros(output_size, boxes_a.options());

    if (unit_flag == UNIT_FLAG_DEGREE) {
        at::Tensor boxes_a_radian = boxes_a.clone().detach();
        boxes_a_radian.index_put_(
            {at::indexing::Slice(), -1},
            boxes_a.index({at::indexing::Slice(), -1}) * PI / 180);
        at::Tensor boxes_b_radian = boxes_b.clone().detach();
        boxes_b_radian.index_put_(
            {at::indexing::Slice(), -1},
            boxes_b.index({at::indexing::Slice(), -1}) * PI / 180);
        EXEC_NPU_CMD(aclnnBoxesOverlapBev, boxes_a_radian, boxes_b_radian, format_flag, clockwise, mode_flag, aligned, margin, res);
        return res;
    }

    if (!aligned && format_flag == FORMAT_FLAG_XYZWHDR && clockwise && unit_flag == UNIT_FLAG_RADIAN &&
            (mode_flag == MODE_FLAG_OVERLAP || mode_flag == MODE_FLAG_IOU)) {
        EXEC_NPU_CMD(aclnnBoxesOverlapBevV1, boxes_a, boxes_b, format_flag, clockwise, mode_flag, aligned, margin, res);
        return res;
    }

    EXEC_NPU_CMD(aclnnBoxesOverlapBev, boxes_a, boxes_b, format_flag, clockwise, mode_flag, aligned, margin, res);
    return res;
}
