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

#ifndef PERCEPTION_FUSED_OPS_CSRC_FUNCTIONS_H_
#define PERCEPTION_FUSED_OPS_CSRC_FUNCTIONS_H_
#include <ATen/Tensor.h>
#include <torch/library.h>

at::Tensor npu_max_pool2d(const at::Tensor& x, int kernel_size, int stride, int padding);

at::Tensor multi_scale_deformable_attn(const at::Tensor& value, const at::Tensor& value_spatial_shapes,
    const at::Tensor& value_level_start_index, const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights);

std::tuple<at::Tensor, at::Tensor, at::Tensor> multi_scale_deformable_attn_backward(const at::Tensor& value,
    const at::Tensor& value_spatial_shapes, const at::Tensor& value_level_start_index,
    const at::Tensor& sampling_locations, const at::Tensor& attention_weights, const at::Tensor& grad_output);

std::tuple<at::Tensor, at::Tensor, at::Tensor> multi_scale_deformable_attn_grad_v2(const at::Tensor& value,
    const at::Tensor& shape, const at::Tensor& level_start_index, const at::Tensor& location_trans,
    const at::Tensor& attn_weight_trans, const at::Tensor& grad_output);

at::Tensor npu_add_relu(at::Tensor& x, const at::Tensor& y);

at::Tensor npu_add_relu_grad(at::Tensor& self, at::Tensor& grad_output);
std::tuple<at::Tensor, at::Tensor> npu_scatter_mean(at::Tensor& src, at::Tensor& index, c10::optional<at::Tensor> out,
    c10::optional<int> dim, c10::optional<int> dim_size);

at::Tensor fused_bias_leaky_relu(
    const at::Tensor& x, const at::Tensor& bias, const double negative_slop, const double scale);

at::Tensor deformable_aggregation(const at::Tensor& mc_ms_feat, const at::Tensor& spatial_shape,
    const at::Tensor& scale_start_index, const at::Tensor& sampling_location, const at::Tensor& weights);
std::tuple<at::Tensor, at::Tensor, at::Tensor> deformable_aggregation_grad(const at::Tensor& mc_ms_feat,
    const at::Tensor& spatial_shape, const at::Tensor& scale_start_index, const at::Tensor& sampling_location,
    const at::Tensor& weights, const at::Tensor& grad_output, const at::Tensor& grad_mc_ms_feat,
    const at::Tensor& grad_sampling_location, const at::Tensor& grad_weights);

std::tuple<at::Tensor, at::Tensor> deformable_conv2d(const at::Tensor& input, const at::Tensor& offset,
    const at::Tensor& weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, int64_t groups, int64_t deformable_groups);

std::tuple<at::Tensor, at::Tensor> modulated_deformable_conv2d(const at::Tensor& input, const at::Tensor& offset,
    const at::Tensor& mask, const at::Tensor& weight, const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, int64_t deformable_groups, int64_t with_bias);

std::tuple<at::Tensor, at::Tensor, at::Tensor> deformable_conv2d_backward(const at::Tensor& input,
    const at::Tensor& weight, const at::Tensor& offset, const at::Tensor& offset_output, const at::Tensor& grad_y,
    at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, int64_t deformable_groups);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> modulated_deformable_conv2d_backward(
    const at::Tensor& input, const at::Tensor& offset, const at::Tensor& mask, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt, const at::Tensor& offset_output, const at::Tensor& grad_y,
    at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, int64_t deformable_groups, int64_t with_bias);

at::Tensor npu_geometric_kernel_attention_func(const at::Tensor& value, const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index, const at::Tensor& sampling_locations, const at::Tensor& attn_weights);

std::tuple<at::Tensor, at::Tensor> npu_geometric_kernel_attention_backward(const at::Tensor& value,
    const at::Tensor& spatial_shapes, const at::Tensor& level_start_index, const at::Tensor& sampling_locations,
    const at::Tensor& attn_weights, const at::Tensor& grad_output);
#endif // PERCEPTION_FUSED_OPS_CSRC_FUNCTIONS_H_
