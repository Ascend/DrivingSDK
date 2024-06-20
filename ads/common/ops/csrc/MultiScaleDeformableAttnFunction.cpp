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

at::Tensor npu_multi_scale_deformable_attn_function(const at::Tensor& value_trans,
    const at::Tensor& value_spatial_shapes, const at::Tensor& value_level_start_index,
    const at::Tensor& sampling_locations_trans, const at::Tensor& attention_weights)
{
    TORCH_CHECK(value_trans.scalar_type() == at::kHalf || value_trans.scalar_type() == at::kFloat,
        "value: float16 or float32 tensor expected but got a tensor with dtype: ", value_trans.scalar_type());
    TORCH_CHECK(value_spatial_shapes.scalar_type() == at::kInt || value_spatial_shapes.scalar_type() == at::kLong,
        "value_spatial_shapes: int32 or int64 tensor expected but got a tensor with dtype: ",
        value_spatial_shapes.scalar_type());
    TORCH_CHECK(value_level_start_index.scalar_type() == at::kInt || value_level_start_index.scalar_type() == at::kLong,
        "value_level_start_index: int32 or int64 tensor expected but got a tensor with dtype: ",
        value_level_start_index.scalar_type());
    TORCH_CHECK(
        sampling_locations_trans.scalar_type() == at::kHalf || sampling_locations_trans.scalar_type() == at::kFloat,
        "sampling_locations: float16 or float32 tensor expected but got a tensor with dtype: ",
        sampling_locations_trans.scalar_type());
    TORCH_CHECK(attention_weights.scalar_type() == at::kHalf || attention_weights.scalar_type() == at::kFloat,
        "attention_weights: float16 or float32 tensor expected but got a tensor with dtype: ",
        attention_weights.scalar_type());

    auto ori_dtype = value_trans.scalar_type();
    // construct the output tensor of the NPU
    auto value_size = value_trans.sizes();
    auto location_size = sampling_locations_trans.sizes();
    auto embed_dims = value_size[3];
    auto output_size = {value_size[0], location_size[1], value_size[1] * embed_dims};

    auto num_points = location_size[5];
    auto num_levels = location_size[3];
    auto data_total = embed_dims + num_points + num_levels;

    TORCH_CHECK(data_total < 512, "data_total is over 512: embed_dims ", embed_dims, ", num_points is ", num_points,
        ", num_level is ", num_levels, ".");
    TORCH_CHECK(embed_dims % 8 == 0, "embed_dims must be a multiple of 8, but embed_dims is ", embed_dims, ".");

    at::Tensor result = at::zeros(output_size, value_trans.options().dtype(at::kFloat));

    at::Tensor value_cp = value_trans.to(at::kFloat);
    at::Tensor value_spatial_shapes_cp = value_spatial_shapes.to(at::kInt);
    at::Tensor value_level_start_index_cp = value_level_start_index.to(at::kInt);
    at::Tensor sampling_locations_cp = sampling_locations_trans.to(at::kFloat);
    at::Tensor attention_weights_cp = attention_weights.to(at::kFloat);

    EXEC_NPU_CMD(aclnnMultiScaleDeformableAttn, value_cp, value_spatial_shapes_cp, value_level_start_index_cp,
        sampling_locations_cp, attention_weights_cp, result);

    return result.to(ori_dtype);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> multi_scale_deformable_attn_grad_v2(const at::Tensor& value_trans,
    const at::Tensor& shape, const at::Tensor& level_start_index, const at::Tensor& location_trans,
    const at::Tensor& attn_weight_trans, const at::Tensor& grad_output)
{
    TORCH_CHECK(value_trans.scalar_type() == at::kHalf || value_trans.scalar_type() == at::kFloat,
        "value_trans: float16 or float32 tensor expected but got a tensor with dtype: ", value_trans.scalar_type());
    TORCH_CHECK(shape.scalar_type() == at::kInt || shape.scalar_type() == at::kLong,
        "spatial_shapes: int32 or int64 tensor expected but got a tensor with dtype: ", shape.scalar_type());
    TORCH_CHECK(level_start_index.scalar_type() == at::kInt || level_start_index.scalar_type() == at::kLong,
        "level_start_index: int32 or int64 tensor expected but got a tensor with dtype: ",
        level_start_index.scalar_type());
    TORCH_CHECK(location_trans.scalar_type() == at::kHalf || location_trans.scalar_type() == at::kFloat,
        "sampling_locations: float16 or float32 tensor expected but got a tensor with dtype: ",
        location_trans.scalar_type());
    TORCH_CHECK(attn_weight_trans.scalar_type() == at::kHalf || attn_weight_trans.scalar_type() == at::kFloat,
        "attn_weight_trans: float16 or float32 tensor expected but got a tensor with dtype: ",
        attn_weight_trans.scalar_type());
    TORCH_CHECK(grad_output.scalar_type() == at::kHalf || grad_output.scalar_type() == at::kFloat,
        "grad_output: float16 or float32 tensor expected but got a tensor with dtype: ", grad_output.scalar_type());

    auto ori_dtype = value_trans.scalar_type();
    auto value_trans_size = value_trans.sizes();
    auto location_trans_size = location_trans.sizes();
    auto attn_weight_trans_size = attn_weight_trans.sizes();
    auto num_heads = value_trans_size[1];
    auto embed_dims = value_trans_size[3];
    auto num_points = location_trans_size[3];
    auto num_levels = location_trans_size[2];
    auto data_total = embed_dims + num_points + num_levels;
    TORCH_CHECK(data_total < 512, "data_total is over 512: embed_dims ", embed_dims, " num_points is ", num_points,
        " num_level is ", num_levels, ".");
    TORCH_CHECK(embed_dims % 8 == 0, "embed_dims must be a multiple of 8, but embed_dims is ", embed_dims, ".");

    at::Tensor grad_value_trans = at::zeros(value_trans_size, value_trans.options().dtype(at::kFloat));
    at::Tensor grad_location_trans = at::zeros(location_trans_size, location_trans.options().dtype(at::kFloat));
    at::Tensor grad_attn_weight_trans =
        at::zeros(attn_weight_trans_size, attn_weight_trans.options().dtype(at::kFloat));

    at::Tensor value_trans_fp = value_trans.to(at::kFloat);
    at::Tensor shape_fp = shape.to(at::kInt);
    at::Tensor level_start_index_fp = level_start_index.to(at::kInt);
    at::Tensor sampling_locations_fp = location_trans.to(at::kFloat);
    at::Tensor attn_weight_fp = attn_weight_trans.to(at::kFloat);
    at::Tensor grad_output_fp = grad_output.to(at::kFloat);
    EXEC_NPU_CMD(aclnnMultiScaleDeformableAttnGradV2, value_trans_fp, shape_fp, level_start_index_fp, sampling_locations_fp,
        attn_weight_fp, grad_output_fp, grad_value_trans, grad_location_trans, grad_attn_weight_trans);
    return std::make_tuple(
        grad_value_trans.to(ori_dtype), grad_location_trans.to(ori_dtype), grad_attn_weight_trans.to(ori_dtype));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> multi_scale_deformable_attn_grad(const at::Tensor& value_trans,
    const at::Tensor& shape, const at::Tensor& level_start_index, const at::Tensor& location_trans,
    const at::Tensor& attn_weight, const at::Tensor& grad_output)
{
    TORCH_CHECK(value_trans.scalar_type() == at::kHalf || value_trans.scalar_type() == at::kFloat,
        "value_trans: float16 or float32 tensor expected but got a tensor with dtype: ", value_trans.scalar_type());
    TORCH_CHECK(shape.scalar_type() == at::kInt || shape.scalar_type() == at::kLong,
        "spatial_shapes: int32 or int64 tensor expected but got a tensor with dtype: ", shape.scalar_type());
    TORCH_CHECK(level_start_index.scalar_type() == at::kInt || level_start_index.scalar_type() == at::kLong,
        "level_start_index: int32 or int64 tensor expected but got a tensor with dtype: ",
        level_start_index.scalar_type());
    TORCH_CHECK(location_trans.scalar_type() == at::kHalf || location_trans.scalar_type() == at::kFloat,
        "sampling_locations: float16 or float32 tensor expected but got a tensor with dtype: ",
        location_trans.scalar_type());
    TORCH_CHECK(attn_weight.scalar_type() == at::kHalf || attn_weight.scalar_type() == at::kFloat,
        "attn_weight: float16 or float32 tensor expected but got a tensor with dtype: ", attn_weight.scalar_type());
    TORCH_CHECK(grad_output.scalar_type() == at::kHalf || grad_output.scalar_type() == at::kFloat,
        "grad_output: float16 or float32 tensor expected but got a tensor with dtype: ", grad_output.scalar_type());

    auto ori_dtype = value_trans.scalar_type();
    auto value_size = value_trans.sizes();
    auto location_size = location_trans.sizes();
    auto attn_weight_size = attn_weight.sizes();
    auto num_heads = value_size[1];
    auto embed_dims = value_size[3];
    auto num_points = location_size[5];
    auto num_levels = location_size[3];
    auto data_total = embed_dims + num_points + num_levels;
    TORCH_CHECK(data_total < 512, "data_total is over 512: embed_dims ", embed_dims, " num_points is ", num_points,
        " num_level is ", num_levels, ".");
    TORCH_CHECK(embed_dims % 8 == 0, "embed_dims must be a multiple of 8, but embed_dims is ", embed_dims, ".");

    at::Tensor grad_value_trans = at::zeros(value_size, value_trans.options().dtype(at::kFloat));
    at::Tensor grad_location_trans = at::zeros(location_size, location_trans.options().dtype(at::kFloat));
    at::Tensor grad_attn_weight = at::zeros(attn_weight_size, attn_weight.options().dtype(at::kFloat));

    at::Tensor value_fp = value_trans.to(at::kFloat);
    at::Tensor shape_fp = shape.to(at::kInt);
    at::Tensor level_start_index_fp = level_start_index.to(at::kInt);
    at::Tensor sampling_locations_fp = location_trans.to(at::kFloat);
    at::Tensor attn_weight_fp = attn_weight.to(at::kFloat);
    at::Tensor grad_output_fp = grad_output.to(at::kFloat);
    EXEC_NPU_CMD(aclnnMultiScaleDeformableAttnGrad, value_fp, shape_fp, level_start_index_fp, sampling_locations_fp,
        attn_weight_fp, grad_output_fp, grad_value_trans, grad_location_trans, grad_attn_weight);
    return std::make_tuple(
        grad_value_trans.to(ori_dtype), grad_location_trans.to(ori_dtype), grad_attn_weight.to(ori_dtype));
}
