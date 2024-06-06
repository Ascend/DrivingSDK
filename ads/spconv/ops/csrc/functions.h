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
#endif // PERCEPTION_VISION_OPS_CSRC_FUNCTIONS_H_
