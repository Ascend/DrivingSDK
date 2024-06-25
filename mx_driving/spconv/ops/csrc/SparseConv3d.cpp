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
#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_sparse_conv3d(const at::Tensor& feature, const at::Tensor& indices, const at::Tensor& weight,
                                                                 at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding,
                                                                 int out_channel, at::IntArrayRef outSpatialShape, int batch_size)
{
    TORCH_CHECK_NPU(feature);
    TORCH_CHECK_NPU(indices);
    TORCH_CHECK_NPU(weight);

    auto indices_size = indices.sizes();
    int64_t kernelsum = 1;
    for (int32_t i = 0; i < kernel_size.size(); i++) {
        kernelsum  *= kernel_size[i];
    }
    int64_t outputsum = indices_size[0] * kernelsum;
    c10::SmallVector<int64_t, 8> output_size = {outputsum, out_channel};
    c10::SmallVector<int64_t, 8> indices_out_size = {outputsum};
    c10::SmallVector<int64_t, 8> indices_pairs_size = {outputsum, indices_size[1]};

    c10::SmallVector<int64_t, 8> spatial_size = {batch_size, outSpatialShape[0], outSpatialShape[1], outSpatialShape[2], out_channel};
    at::IntArrayRef outputShape =  at::IntArrayRef(spatial_size);

    at::Tensor weight_trans = weight.transpose(-1, -2).contiguous();

    at::Tensor out = at::empty(output_size, feature.options());
    at::Tensor indices_out = at::empty(indices_out_size, indices.options()).fill_(-1);
    at::Tensor indices_pairs = at::empty(indices_pairs_size, indices.options()).fill_(-1);
    EXEC_NPU_CMD(aclnnSparseConv3d, feature, indices, weight_trans, outputShape,
                 stride, padding, out, indices_out, indices_pairs);
    return std::tie(out, indices_pairs, indices_out);
}