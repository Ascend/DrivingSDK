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
at::Tensor& rotated_iou_npu_nocheck(at::Tensor& iou, const at::Tensor& boxes, const at::Tensor& query_boxes, bool trans,
    int64_t mode, bool is_cross, double v_threshold, double e_threshold)
{
    string mode_str = (mode == 0) ? "iou" : "iof";

    at_npu::native::OpCommand cmd;
    cmd.Name("RotatedIou")
        .Input(boxes)
        .Input(query_boxes)
        .Output(iou)
        .Attr("trans", trans)
        .Attr("mode", mode_str)
        .Attr("is_cross", is_cross)
        .Attr("value", static_cast<float>(v_threshold))
        .Attr("value", static_cast<float>(e_threshold))
        .Run();
    return iou;
}
} // namespace

at::Tensor npu_rotated_iou(const at::Tensor& boxes, const at::Tensor& query_boxes, bool trans, int64_t mode,
    bool is_cross, double v_threshold, double e_threshold)
{
    TORCH_CHECK(boxes.ndimension() == 3 && query_boxes.ndimension() == 3);

    auto origin_dtype = boxes.scalar_type();

    at::Tensor boxes_cp = boxes.permute({0, 2, 1});
    if (origin_dtype == at::kHalf) {
        boxes_cp = boxes_cp.to(at::kFloat);
    }
    at::Tensor query_boxes_cp = query_boxes.permute({0, 2, 1});
    if (query_boxes_cp.scalar_type() == at::kHalf) {
        query_boxes_cp = query_boxes_cp.to(at::kFloat);
    }

    int64_t B = boxes_cp.size(0);
    int64_t N = boxes_cp.size(-1);
    int64_t K = query_boxes_cp.size(-1);

    c10::SmallVector<int64_t, 8U> output_size({B, N, K});
    at::Tensor iou = at::empty(output_size, boxes_cp.options());

    rotated_iou_npu_nocheck(iou, boxes_cp, query_boxes_cp, trans, mode, is_cross, v_threshold, e_threshold);
    iou = iou.to(origin_dtype);
    return iou;
}
