// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/framework/OpCommand.h"
#include "common.h"

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_batch_nms(
    const at::Tensor& self,
    const at::Tensor& scores,
    double score_threshold,
    double iou_threshold,
    int64_t max_size_per_class,
    int64_t max_total_size,
    bool change_coordinate_frame,
    bool transpose_box)
{
    at::Tensor nmsed_boxes = at::empty({self.size(0), max_total_size, 4}, self.options());
    at::Tensor nmsed_scores = at::empty({self.size(0), max_total_size}, self.options());
    at::Tensor nmsed_classes = at::empty({self.size(0), max_total_size}, self.options());
    at::Tensor nmsed_num = at::empty({self.size(0)}, self.options().dtype(at::kInt));
    at_npu::native::OpCommand cmd;
    cmd.Name("BatchMultiClassNonMaxSuppression")
        .Input(self)
        .Input(scores)
        .Output(nmsed_boxes)
        .Output(nmsed_scores)
        .Output(nmsed_classes)
        .Output(nmsed_num)
        .Attr("score_threshold", static_cast<float>(score_threshold))
        .Attr("iou_threshold", static_cast<float>(iou_threshold))
        .Attr("max_size_per_class", max_size_per_class)
        .Attr("max_total_size", max_total_size)
        .Attr("change_coordinate_frame", change_coordinate_frame)
        .Attr("transpose_box", transpose_box)
        .Run();
    return std::tie(nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num);
}
