// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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
#ifndef COMMON_OPS_CSRC_FUNCTIONS_H_
#define COMMON_OPS_CSRC_FUNCTIONS_H_

#include <ATen/Tensor.h>
#include <ATen/ATen.h>
#include <ATen/core/Scalar.h>
#include <torch/extension.h>
#include <torch/library.h>

std::tuple<at::Tensor, at::Tensor> npu_scatter_max(const at::Tensor& updates, const at::Tensor& indices, c10::optional<at::Tensor> out);
at::Tensor npu_scatter_max_backward(const at::Tensor& x, const at::Tensor& segment_ids, const at::Tensor& num_segments);

at::Tensor npu_rotated_box_decode(const at::Tensor &self, const at::Tensor &deltas, const at::Tensor &weight);
at::Tensor npu_rotated_box_encode(
    const at::Tensor& self,
    const at::Tensor& gtBox,
    const at::Tensor& weight);
at::Tensor npu_rotated_iou(
    const at::Tensor& boxes,
    const at::Tensor& query_boxes,
    bool trans,
    int64_t mode,
    bool is_cross,
    double v_threshold,
    double e_threshold);
at::Tensor npu_rotated_overlaps(
    const at::Tensor& self,
    const at::Tensor& query_boxes,
    bool trans);
at::Tensor npu_scatter(const at::Tensor& self, const at::Tensor& indices, const at::Tensor& updates, int64_t dim);
at::Tensor npu_sign_bits_pack(const at::Tensor& self, int64_t size);
at::Tensor npu_sign_bits_unpack(py::args args);
at::Tensor npu_softmax_cross_entropy_with_logits(const at::Tensor &self, const at::Tensor &lables);
at::Tensor npu_stride_add(py::args args);
at::Tensor npu_transpose(const at::Tensor &self, at::IntArrayRef perm, bool require_contiguous);
at::Tensor npu_yolo_boxes_encode(
    const at::Tensor& anchor_boxes,
    const at::Tensor& gt_bboxes,
    const at::Tensor& stride,
    bool performance_mode);
at::Tensor npu_scatter(const at::Tensor& self, const at::Tensor& indices, const at::Tensor& updates, int64_t dim);
at::Tensor npu_rotary_mul(const at::Tensor &self, const at::Tensor &r1, const at::Tensor &r2);
at::Tensor npu_silu(const at::Tensor& self);
at::Tensor& npu_silu_(at::Tensor& self);
at::Tensor npu_abs(const at::Tensor& self);
at::Tensor npu_fast_gelu_backward(const at::Tensor& grad, const at::Tensor& self);
at::Tensor npu_fast_gelu(const at::Tensor& self);
at::Tensor npu_anchor_response_flags(const at::Tensor& self, at::IntArrayRef featmap_size, at::IntArrayRef stride, int64_t num_base_anchors);
at::Tensor npu_bounding_box_decode(
    const at::Tensor& rois,
    const at::Tensor& deltas,
    double means0,
    double means1,
    double means2,
    double means3,
    double stds0,
    double stds1,
    double stds2,
    double stds3,
    at::IntArrayRef max_shape,
    double wh_ratio_clip);
at::Tensor npu_bounding_box_encode(
    const at::Tensor& anchor_box,
    const at::Tensor& ground_truth_box,
    double means0,
    double means1,
    double means2,
    double means3,
    double stds0,
    double stds1,
    double stds2,
    double stds3);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_batch_nms(
    const at::Tensor& self,
    const at::Tensor& scores,
    double score_threshold,
    double iou_threshold,
    int64_t max_size_per_class,
    int64_t max_total_size,
    bool change_coordinate_frame,
    bool transpose_box);
at::Tensor npu_confusion_transpose(
    const at::Tensor& self,
    at::IntArrayRef perm,
    at::IntArrayRef shape,
    bool transpose_first);
at::Tensor npu_confusion_transpose_backward(
    const at::Tensor& grad,
    at::IntArrayRef perm,
    at::IntArrayRef shape,
    bool transpose_first);
at::Tensor npu_conv_transpose2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups);

at::Tensor furthest_point_sampling_with_dist(const at::Tensor &points_dist, const at::Tensor &nearest_temp, const int32_t num_points);

at::Tensor npu_broadcast(const at::Tensor& self, at::IntArrayRef size);
at::Tensor& npu_broadcast_out(const at::Tensor& self, at::IntArrayRef size, at::Tensor& result);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_dynamic_scatter(
    const at::Tensor &feats,
    const at::Tensor &coors,
    int64_t reduce_type);
at::Tensor npu_moe_tutel(
    const at::Tensor &self,
    const at::Tensor &gates,
    const at::Tensor &indices,
    const at::Tensor &locations,
    int64_t capacity);
at::Tensor npu_moe_tutel_data_backward(
    const at::Tensor &y_grad,
    const at::Tensor &gates,
    const at::Tensor &indices,
    const at::Tensor &locations);
at::Tensor npu_moe_tutel_gate_backward(
    const at::Tensor &self,
    const at::Tensor &y_grad,
    const at::Tensor &indices,
    const at::Tensor &locations);
at::Tensor npu_multi_scale_deformable_attn_function(const at::Tensor& value,
                                                    const at::Tensor& value_spatial_shapes,
                                                    const at::Tensor& value_level_start_index,
                                                    const at::Tensor& sampling_locations,
                                                    const at::Tensor& attention_weights);
std::tuple<at::Tensor, at::Tensor, at::Tensor> multi_scale_deformable_attn_grad(const at::Tensor& value, const at::Tensor& shape,
    const at::Tensor& level_start_index, const at::Tensor& location, const at::Tensor& attn_weight, const at::Tensor& grad_output);
at::Tensor npu_ads_add(const at::Tensor &tensor1, const at::Tensor &tensor2);

at::Tensor DynamicVoxelization(
    const at::Tensor &points,
    at::Tensor &coors,
    const int gridX,
    const int gridY,
    const int gridZ,
    const double voxelX,
    const double voxelY,
    const double voxelZ,
    const double coorsMinX,
    const double coorsMinY,
    const double coorsMinZ);
#endif // COMMON_OPS_CSRC_FUNCTIONS_H_
