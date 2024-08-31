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

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_subm_sparse_conv3d(const at::Tensor& feature, const at::Tensor& indices,
                                                                      const at::Tensor& weight,
                                                                      at::IntArrayRef kernel_size, int out_channel,
                                                                      at::IntArrayRef outSpatialShape, int batch_size);

std::tuple<at::Tensor, at::Tensor> multi_to_sparse(const at::Tensor& out_features, const at::Tensor& unique_indices_offset,
                                                   const at::Tensor& sorted_idx_to_former_indices, const at::Tensor& outidx_pair);

std::tuple<at::Tensor, at::Tensor> multi_to_sparse_v2(const at::Tensor& features, const at::Tensor& weight, const at::Tensor& unique_indices_offset,
                                                      const at::Tensor& sorted_idx_to_former_indices, const at::Tensor& outidx_pair);

std::tuple<at::Tensor, at::Tensor> npu_sparse_conv3d(const at::Tensor& indices, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding,
                                                     int out_channel, at::IntArrayRef outSpatialShape, int batch_size);

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_sparse_inverse_conv3d(const at::Tensor& feature, const at::Tensor& indices, const at::Tensor& weight,
                                                                         at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding,
                                                                         at::IntArrayRef dilation, at::IntArrayRef output_padding,
                                                                         int out_channel, at::IntArrayRef outSpatialShape, int batch_size);

std::tuple<at::Tensor, at::Tensor> npu_sparse_conv3d_grad(const at::Tensor& indices_offset, const at::Tensor& former_sorted_indices,
                                                          const at::Tensor& feature, const at::Tensor& weight, const at::Tensor& grad);

#endif // PERCEPTION_VISION_OPS_CSRC_FUNCTIONS_H_
