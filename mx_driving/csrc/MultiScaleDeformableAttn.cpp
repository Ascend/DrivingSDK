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
constexpr size_t BATCH_IDX = 0;
constexpr size_t QUERY_IDX = 1;
constexpr size_t HEAD_IDX = 2;
constexpr size_t EMBED_IDX = 3;
constexpr size_t LEVEL_IDX = 3;
constexpr size_t POINT_IDX = 4;
} // namespace

at::Tensor multi_scale_deformable_attn(const at::Tensor& value, const at::Tensor& value_spatial_shapes,
    const at::Tensor& value_level_start_index, const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights)
{
    TORCH_CHECK(value.scalar_type() == at::kHalf || value.scalar_type() == at::kFloat,
        "value: float16 or float32 tensor expected but got a tensor with dtype: ", value.scalar_type());
    TORCH_CHECK(value_spatial_shapes.scalar_type() == at::kInt,
        "value_spatial_shapes: int32 tensor expected but got a tensor with dtype: ",
        value_spatial_shapes.scalar_type());
    TORCH_CHECK(value_level_start_index.scalar_type() == at::kInt,
        "value_level_start_index: int32 tensor expected but got a tensor with dtype: ",
        value_level_start_index.scalar_type());
    TORCH_CHECK(sampling_locations.scalar_type() == at::kHalf || sampling_locations.scalar_type() == at::kFloat,
        "sampling_locations: float16 or float32 tensor expected but got a tensor with dtype: ",
        sampling_locations.scalar_type());
    TORCH_CHECK(attention_weights.scalar_type() == at::kHalf || attention_weights.scalar_type() == at::kFloat,
        "attention_weights: float16 or float32 tensor expected but got a tensor with dtype: ",
        attention_weights.scalar_type());
    TORCH_CHECK(value.size(EMBED_IDX) <= 64, "The number of embedding dimensions should be less than or equal to 64");
    TORCH_CHECK(sampling_locations.size(LEVEL_IDX) * sampling_locations.size(POINT_IDX) <= 64,
        "The product of the number of levels and the number of points should be less than or equal to 64");

    at::SmallVector<int64_t, 4> output_size = {sampling_locations.size(BATCH_IDX), sampling_locations.size(QUERY_IDX),
        value.size(HEAD_IDX) * value.size(EMBED_IDX)};
    at::Tensor output = at::empty(output_size, value.options().dtype(at::kFloat));

    if (ASCEND_UNLIKELY(value.scalar_type() == at::kHalf)) {
        at::Tensor value_fp32 = value.to(at::kFloat);
        at::Tensor sampling_locations_fp32 = sampling_locations.to(at::kFloat);
        at::Tensor attention_weights_fp32 = attention_weights.to(at::kFloat);
        EXEC_NPU_CMD(aclnnMultiScaleDeformableAttn, value_fp32, value_spatial_shapes, value_level_start_index,
            sampling_locations_fp32, attention_weights_fp32, output);
        return output.to(at::kHalf);
    }

    EXEC_NPU_CMD(aclnnMultiScaleDeformableAttn, value, value_spatial_shapes, value_level_start_index,
        sampling_locations, attention_weights, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> multi_scale_deformable_attn_backward(const at::Tensor& value,
    const at::Tensor& value_spatial_shapes, const at::Tensor& value_level_start_index,
    const at::Tensor& sampling_locations, const at::Tensor& attention_weights, const at::Tensor& grad_output)
{
    TORCH_CHECK(value.scalar_type() == at::kHalf || value.scalar_type() == at::kFloat,
        "value: float16 or float32 tensor expected but got a tensor with dtype: ", value.scalar_type());
    TORCH_CHECK(value_spatial_shapes.scalar_type() == at::kInt,
        "value_spatial_shapes: int32 or int64 tensor expected but got a tensor with dtype: ",
        value_spatial_shapes.scalar_type());
    TORCH_CHECK(value_level_start_index.scalar_type() == at::kInt,
        "value_level_start_index: int32 or int64 tensor expected but got a tensor with dtype: ",
        value_level_start_index.scalar_type());
    TORCH_CHECK(sampling_locations.scalar_type() == at::kHalf || sampling_locations.scalar_type() == at::kFloat,
        "sampling_locations: float16 or float32 tensor expected but got a tensor with dtype: ",
        sampling_locations.scalar_type());
    TORCH_CHECK(attention_weights.scalar_type() == at::kHalf || attention_weights.scalar_type() == at::kFloat,
        "attn_weight_trans: float16 or float32 tensor expected but got a tensor with dtype: ",
        attention_weights.scalar_type());
    TORCH_CHECK(grad_output.scalar_type() == at::kHalf || grad_output.scalar_type() == at::kFloat,
        "grad_output: float16 or float32 tensor expected but got a tensor with dtype: ", grad_output.scalar_type());

    TORCH_CHECK(value.size(EMBED_IDX) <= 64, "The number of embedding dimensions should be less than or equal to 64");
    TORCH_CHECK(sampling_locations.size(LEVEL_IDX) * sampling_locations.size(POINT_IDX) <= 64,
        "The product of the number of levels and the number of points should be less than or equal to 64");

    at::Tensor grad_value = at::zeros_like(value, value.options().dtype(at::kFloat));
    at::Tensor grad_sampling_loc = at::empty_like(sampling_locations, sampling_locations.options().dtype(at::kFloat));
    at::Tensor grad_attn_weight = at::empty_like(attention_weights, attention_weights.options().dtype(at::kFloat));

    // Check if the number of spatial shapes does not match the number of attention weights
    if (ASCEND_UNLIKELY(value_spatial_shapes.size(0) != attention_weights.size(LEVEL_IDX))) {
        grad_sampling_loc.zero_();
        grad_attn_weight.zero_();
    }

    if (ASCEND_UNLIKELY(value.scalar_type() == at::kHalf)) {
        at::Tensor grad_value_fp32 = grad_value.to(at::kFloat);
        at::Tensor value_fp32 = value.to(at::kFloat);
        at::Tensor sampling_locations_fp32 = sampling_locations.to(at::kFloat);
        at::Tensor attention_weights_fp32 = attention_weights.to(at::kFloat);
        EXEC_NPU_CMD(aclnnMultiScaleDeformableAttnGrad, value_fp32, value_spatial_shapes, value_level_start_index,
            sampling_locations_fp32, attention_weights_fp32, grad_value_fp32, grad_value, grad_sampling_loc,
            grad_attn_weight);
        return std::make_tuple(
            grad_value.to(at::kHalf), grad_sampling_loc.to(at::kHalf), grad_attn_weight.to(at::kHalf));
    }

    EXEC_NPU_CMD(aclnnMultiScaleDeformableAttnGrad, value, value_spatial_shapes, value_level_start_index,
        sampling_locations, attention_weights, grad_output, grad_value, grad_sampling_loc, grad_attn_weight);
    return std::make_tuple(grad_value, grad_sampling_loc, grad_attn_weight);
}
