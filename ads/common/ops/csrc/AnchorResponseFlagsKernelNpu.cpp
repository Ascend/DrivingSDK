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

namespace {
c10::SmallVector<int64_t, SIZE> infersize_npu_anchor_response_flags(
    at::IntArrayRef featmap_size,
    int64_t num_base_anchors)
{
    int64_t output_value = featmap_size[0] * featmap_size[1] * num_base_anchors;
    c10::SmallVector<int64_t, SIZE> output_size = {output_value};
    return output_size;
}

inline void anchor_response_flags_check(
    const at::Tensor& self,
    at::IntArrayRef featmap_size,
    at::IntArrayRef stride)
{
    TORCH_CHECK(
        featmap_size.size() == 2,
        "expected feat_map_size equals to 2, but got size ",
        featmap_size.size());
    TORCH_CHECK(
        self.dim() == 2 && self.size(1) == 4,
        "Non-empty 2D gt_bboxes tensor expected but got a tensor with sizes ",
        self.sizes());
    TORCH_CHECK(
        self.scalar_type() == at::kHalf || self.scalar_type() == at::kFloat,
        "float16 or float32 tensor expected but got a tensor with dtype: ",
        self.scalar_type());
}
} // namespace

at::Tensor npu_anchor_response_flags(
    const at::Tensor& self,
    at::IntArrayRef featmap_size,
    at::IntArrayRef stride,
    int64_t num_base_anchors)
{
    anchor_response_flags_check(self, featmap_size, stride);
    auto output_size = infersize_npu_anchor_response_flags(featmap_size, num_base_anchors);
    auto options = self.options().dtype(at::kByte);
    at::Tensor result = at::empty(output_size, options);

    at::Tensor self_cp = self.to(at::kFloat);

    at_npu::native::OpCommand cmd;
    cmd.Name("AnchorResponseFlags")
        .Input(self_cp)
        .Output(result)
        .Attr("featmap_size", featmap_size)
        .Attr("strides", stride)
        .Attr("num_base_anchors", num_base_anchors)
        .Run();
    return result;
}
