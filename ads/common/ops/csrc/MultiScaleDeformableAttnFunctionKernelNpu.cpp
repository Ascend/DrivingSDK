#include <ATen/ATen.h>
#include "OpApiCommon.h"
#include "functions.h"

at::Tensor npu_multi_scale_deformable_attn_function(const at::Tensor& value,
                                                    const at::Tensor& value_spatial_shapes,
                                                    const at::Tensor& value_level_start_index,
                                                    const at::Tensor& sampling_locations,
                                                    const at::Tensor& attention_weights)
{
    TORCH_CHECK(
        value.scalar_type() == at::kHalf || value.scalar_type() == at::kFloat,
        "value: float16 or float32 tensor expected but got a tensor with dtype: ",
        value.scalar_type());
    TORCH_CHECK(
        value_spatial_shapes.scalar_type() == at::kInt || value_spatial_shapes.scalar_type() == at::kLong,
        "value_spatial_shapes: int32 or int64 tensor expected but got a tensor with dtype: ",
        value_spatial_shapes.scalar_type());
    TORCH_CHECK(
        value_level_start_index.scalar_type() == at::kInt || value_level_start_index.scalar_type() == at::kLong,
        "value_level_start_index: int32 or int64 tensor expected but got a tensor with dtype: ",
        value_level_start_index.scalar_type());
    TORCH_CHECK(
        sampling_locations.scalar_type() == at::kHalf || sampling_locations.scalar_type() == at::kFloat,
        "sampling_locations: float16 or float32 tensor expected but got a tensor with dtype: ",
        sampling_locations.scalar_type());
    TORCH_CHECK(
        attention_weights.scalar_type() == at::kHalf || attention_weights.scalar_type() == at::kFloat,
        "attention_weights: float16 or float32 tensor expected but got a tensor with dtype: ",
        attention_weights.scalar_type());

    auto ori_dtype = value.scalar_type();
    // construct the output tensor of the NPU
    auto value_size = value.sizes();
    auto location_size = sampling_locations.sizes();
    auto output_size = {value_size[0], location_size[1], value_size[2] * value_size[3]};

    at::Tensor result = at::empty(output_size, value.options().dtype(at::kFloat));

    // reset inputs
    at::Tensor value_cp = value.to(at::kFloat);
    at::Tensor value_spatial_shapes_cp = value_spatial_shapes.to(at::kInt);
    at::Tensor value_level_start_index_cp = value_level_start_index.to(at::kInt);
    at::Tensor sampling_locations_cp = sampling_locations.to(at::kFloat);
    at::Tensor attention_weights_cp = attention_weights.to(at::kFloat);

    EXEC_NPU_CMD(aclnnMultiScaleDeformableAttnFunction, value_cp, value_spatial_shapes_cp,
                 value_level_start_index_cp, sampling_locations_cp,
                 attention_weights_cp, result);

    return result.to(ori_dtype);
}
