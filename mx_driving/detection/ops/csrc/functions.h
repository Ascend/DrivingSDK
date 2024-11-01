// Copyright (c) 2024, Huawei Technologies.All rights reserved.
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
#ifndef PERCEPTION_VISION_OPS_CSRC_FUNCTIONS_H_
#define PERCEPTION_VISION_OPS_CSRC_FUNCTIONS_H_

#include <ATen/ATen.h>
#include <torch/library.h>

std::tuple<at::Tensor, at::Tensor> nms3d_normal(const at::Tensor& boxes, double nms_overlap_thresh);

std::tuple<at::Tensor, at::Tensor> nms3d(const at::Tensor& boxes, double threshold);

at::Tensor npu_rotated_overlaps(const at::Tensor& self, const at::Tensor& query_boxes, bool trans);

at::Tensor npu_rotated_iou(const at::Tensor& boxes, const at::Tensor& query_boxes, bool trans, int64_t mode,
    bool is_cross, double v_threshold, double e_threshold);

at::Tensor npu_boxes_overlap_bev(const at::Tensor &boxes_a, const at::Tensor &boxes_b);

void roi_align_rotated_v2_forward_npu(const at::Tensor& input, const at::Tensor& rois_map, at::Tensor& output,
                                      double spatial_scale, int32_t sampling_ratio, int32_t pooled_height, int32_t pooled_width,
                                      bool aligned, bool clockwise);
at::Tensor npu_roi_align_rotated_grad_v2(const at::Tensor& input,
                                         const at::Tensor& rois, const at::Tensor& grad_output,
                                         int32_t pooled_height, int32_t pooled_width, double spatial_scale,
                                         int32_t sampling_ratio, bool aligned, bool clockwise);

at::Tensor npu_box_iou_quadri(const at::Tensor &boxes_a, const at::Tensor &boxes_b,
    const int64_t mode_flag, const bool aligned);

at::Tensor npu_box_iou_rotated(const at::Tensor &boxes_a, const at::Tensor &boxes_b,
    const int64_t mode_flag, const bool aligned);

void border_align_forward_npu(const at::Tensor& input, const at::Tensor& rois, at::Tensor& output, const int32_t pooled_size);

at::Tensor border_align_backward(const at::Tensor& grad_out, const at::Tensor& boxes, const at::Tensor& argmax_idx,
                                 int32_t pool_size, int32_t height, int32_t width);
#endif // PERCEPTION_VISION_OPS_CSRC_FUNCTIONS_H_
