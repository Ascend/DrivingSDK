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
#include "csrc/utils.h"
#include "csrc/functions.h"

constexpr size_t VALUE_BATCH_SIZE_DIM = 0;
constexpr size_t VALUE_NUM_KEYS_DIM = 1;
constexpr size_t VALUE_NUM_HEADS_DIM = 2;
constexpr size_t VALUE_EMBED_DIMS_DIM = 3;
constexpr size_t ATTN_WEIGHTS_BATCH_SIZE_DIM = 0;
constexpr size_t ATTN_WEIGHTS_NUM_QUERIES_DIM = 1;
constexpr size_t ATTN_WEIGHTS_NUM_HEADS_DIM = 2;
constexpr size_t ATTN_WEIGHTS_NUM_LEVELS_DIM = 3;
constexpr size_t ATTN_WEIGHTS_NUM_POINTS_DIM = 4;
constexpr size_t FLOAT32_BYTES = 4;
constexpr size_t BLOCK_BYTES = 32;

void geometric_kernel_attention_forward(const at::Tensor& value_map, const at::Tensor& spatial_shapes, const at::Tensor& level_start_index,
                                        const at::Tensor& sampling_locations, const at::Tensor& attention_weights, at::Tensor& output)
{
    TORCH_CHECK(value_map.scalar_type() == at::kHalf || value_map.scalar_type() == at::kFloat,
        "value_map: float16 or float32 tensor expected but got a tensor with dtype: ", value_map.scalar_type());
    TORCH_CHECK(spatial_shapes.scalar_type() == at::kInt || spatial_shapes.scalar_type() == at::kLong,
        "spatial_spatial_shapes: int32 or int64 tensor expected but got a tensor with dtype: ",
        spatial_shapes.scalar_type());
    TORCH_CHECK(level_start_index.scalar_type() == at::kInt || level_start_index.scalar_type() == at::kLong,
        "level_start_index: int32 or int64 tensor expected but got a tensor with dtype: ",
        level_start_index.scalar_type());
    TORCH_CHECK(sampling_locations.scalar_type() == at::kHalf || sampling_locations.scalar_type() == at::kFloat ||
                sampling_locations.scalar_type() == at::kInt || sampling_locations.scalar_type() == at::kLong,
        "sampling_locations: float16, float32, int32 or int64 tensor expected but got a tensor with dtype: ",
        sampling_locations.scalar_type());
    TORCH_CHECK(attention_weights.scalar_type() == at::kHalf || attention_weights.scalar_type() == at::kFloat,
        "attention_weights: float16 or float32 tensor expected but got a tensor with dtype: ",
        attention_weights.scalar_type());

    at::Tensor value = value_map.permute({0, 2, 1, 3}).contiguous();
    EXEC_NPU_CMD(aclnnGeometricKernelAttention, value, spatial_shapes, level_start_index, sampling_locations, attention_weights, output);
}

std::tuple<at::Tensor, at::Tensor> geometric_kernel_attention_backward(const at::Tensor& value,
    const at::Tensor& spatial_shapes, const at::Tensor& level_start_index, const at::Tensor& sampling_locations,
    const at::Tensor& attn_weights, const at::Tensor& grad_output)
{
    TORCH_CHECK(value.scalar_type() == at::kHalf || value.scalar_type() == at::kFloat,
        "value: float16 or float32 tensor expected but got a tensor with dtype: ", value.scalar_type());
    TORCH_CHECK(spatial_shapes.scalar_type() == at::kInt || spatial_shapes.scalar_type() == at::kLong,
        "spatial_spatial_shapes: int32 or int64 tensor expected but got a tensor with dtype: ",
        spatial_shapes.scalar_type());
    TORCH_CHECK(level_start_index.scalar_type() == at::kInt || level_start_index.scalar_type() == at::kLong,
        "level_start_index: int32 or int64 tensor expected but got a tensor with dtype: ",
        level_start_index.scalar_type());
    TORCH_CHECK(sampling_locations.scalar_type() == at::kHalf || sampling_locations.scalar_type() == at::kFloat ||
                sampling_locations.scalar_type() == at::kInt || sampling_locations.scalar_type() == at::kLong,
        "sampling_locations: float16, float32, int32 or int64 tensor expected but got a tensor with dtype: ",
        sampling_locations.scalar_type());
    TORCH_CHECK(attn_weights.scalar_type() == at::kHalf || attn_weights.scalar_type() == at::kFloat,
        "attn_weights: float16 or float32 tensor expected but got a tensor with dtype: ",
        attn_weights.scalar_type());
    TORCH_CHECK(grad_output.scalar_type() == at::kHalf || grad_output.scalar_type() == at::kFloat,
        "grad_output: float16 or float32 tensor expected but got a tensor with dtype: ", grad_output.scalar_type());

    auto ori_dtype = value.scalar_type();
    auto value_size = value.sizes();
    auto attn_weights_size = attn_weights.sizes();
    
    auto bs = value_size[VALUE_BATCH_SIZE_DIM];
    auto num_keys = value_size[VALUE_NUM_KEYS_DIM];
    auto num_heads = value_size[VALUE_NUM_HEADS_DIM];
    auto embed_dims = value_size[VALUE_EMBED_DIMS_DIM];
    auto num_queries = attn_weights_size[ATTN_WEIGHTS_NUM_QUERIES_DIM];
    auto num_levels = attn_weights_size[ATTN_WEIGHTS_NUM_LEVELS_DIM];
    auto num_points = attn_weights_size[ATTN_WEIGHTS_NUM_POINTS_DIM];

    TORCH_CHECK(embed_dims % 8 == 0, "embed_dims must be a multiple of 8, but embed_dims is ", embed_dims, ".");

    at::Tensor grad_value = at::zeros({bs, num_keys, num_heads, embed_dims}, value.options().dtype(at::kFloat));
    at::Tensor grad_attn_weights = at::empty({bs, num_queries, num_heads, num_levels, num_points}, attn_weights.options().dtype(at::kFloat));

    at::Tensor value_fp = value.to(at::kFloat);
    at::Tensor spatial_shapes_fp = spatial_shapes.to(at::kInt);
    at::Tensor level_start_index_fp = level_start_index.to(at::kInt);
    at::Tensor sampling_locations_fp = sampling_locations.to(at::kFloat);
    at::Tensor attn_weights_fp = attn_weights.to(at::kFloat);
    at::Tensor grad_output_fp = grad_output.to(at::kFloat);

    EXEC_NPU_CMD(aclnnGeometricKernelAttnGrad, value_fp, spatial_shapes_fp, level_start_index_fp, sampling_locations_fp,
        attn_weights_fp, grad_output_fp, grad_value, grad_attn_weights);

    return std::make_tuple(grad_value.to(ori_dtype), grad_attn_weights.to(ori_dtype));
}
