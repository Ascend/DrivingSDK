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

std::tuple<at::Tensor, at::Tensor> npu_prepare_subm_conv3d(
    const at::Tensor& flattenIndices, at::IntArrayRef outSpatialShape, int batch_size)
{
    int64_t outputnum = 1;
    for (int32_t i = 0; i < outSpatialShape.size(); i++) {
        outputnum *= outSpatialShape[i];
    }
    c10::SmallVector<int64_t, 8> output_size = {batch_size * outputnum};
    auto temp = at::empty(output_size, flattenIndices.options().dtype(at::kFloat)).fill_(-1);
    auto hh2 = at::arange(0, flattenIndices.sizes()[0], flattenIndices.options().dtype(at::kFloat));
    return std::tie(temp, hh2);
}
