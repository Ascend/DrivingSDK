#include <ATen/ATen.h>
#include "csrc/OpApiCommon.h"
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

    auto embed_dims = value_size[3];
    auto num_points = location_size[4];
    auto num_levels = location_size[3];
    auto data_total = embed_dims + num_points + num_levels;

    TORCH_CHECK(
        data_total < 512,
        "data_total is over 512: embed_dims ", embed_dims, " num_points is ", num_points, " num_level is ", num_levels, "." );

    at::Tensor result = at::empty(output_size, value.options().dtype(at::kFloat));

    // reset inputs
    at::Tensor value_cp = value.to(at::kFloat);
    at::Tensor value_spatial_shapes_cp = value_spatial_shapes.to(at::kInt);
    at::Tensor value_level_start_index_cp = value_level_start_index.to(at::kInt);
    at::Tensor sampling_locations_cp = sampling_locations.to(at::kFloat);
    at::Tensor attention_weights_cp = attention_weights.to(at::kFloat);

    EXEC_NPU_CMD(aclnnMultiScaleDeformableAttnFunctionV2, value_cp, value_spatial_shapes_cp,
                 value_level_start_index_cp, sampling_locations_cp,
                 attention_weights_cp, result);

    return result.to(ori_dtype);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> multi_scale_deformable_attn_grad(const at::Tensor& value, const at::Tensor& shape,
                                                                                const at::Tensor& level_start_index,
                                                                                const at::Tensor& location, const at::Tensor& attn_weight,
                                                                                const at::Tensor& grad_output)
{
    TORCH_CHECK(
        value.scalar_type() == at::kHalf || value.scalar_type() == at::kFloat,
        "value: float16 or float32 tensor expected but got a tensor with dtype: ",
        value.scalar_type());
    TORCH_CHECK(
        shape.scalar_type() == at::kInt || shape.scalar_type() == at::kLong,
        "spatial_shapes: int32 or int64 tensor expected but got a tensor with dtype: ",
        shape.scalar_type());
    TORCH_CHECK(
        level_start_index.scalar_type() == at::kInt || level_start_index.scalar_type() == at::kLong,
        "level_start_index: int32 or int64 tensor expected but got a tensor with dtype: ",
        level_start_index.scalar_type());
    TORCH_CHECK(
        location.scalar_type() == at::kHalf || location.scalar_type() == at::kFloat,
        "sampling_locations: float16 or float32 tensor expected but got a tensor with dtype: ",
        location.scalar_type());
    TORCH_CHECK(
        attn_weight.scalar_type() == at::kHalf || attn_weight.scalar_type() == at::kFloat,
        "attn_weight: float16 or float32 tensor expected but got a tensor with dtype: ",
        attn_weight.scalar_type());
    TORCH_CHECK(
        grad_output.scalar_type() == at::kHalf || grad_output.scalar_type() == at::kFloat,
        "grad_output: float16 or float32 tensor expected but got a tensor with dtype: ",
        grad_output.scalar_type());

    auto ori_dtype = value.scalar_type();
    auto value_size = value.sizes();
    auto location_size = location.sizes();
    auto channels = value_size[3];
    auto num_points = location_size[4];
    auto num_levels = location_size[3];
    auto data_total = channels + num_points + num_levels;
    TORCH_CHECK(data_total < 512, "data_total is over 512: channels ", channels, " num_points is ",
                num_points, " num_level is ", num_levels, ".");
    TORCH_CHECK(channels % 8 == 0, "channels must be a multiple of eight, but channels is", channels, ".");
    auto grad_value_size = {value_size[0], value_size[1], value_size[2], value_size[3]};
    auto grad_atten_weight_size = {location_size[0], location_size[1], location_size[2], location_size[3], location_size[4]};
    auto grad_sample_loc_size = {location_size[0], location_size[1], location_size[2], location_size[3], location_size[5], location_size[4]};
    at::Tensor location1 = location.transpose(4, 5).contiguous();
    at::Tensor result1 = at::zeros(grad_value_size, value.options().dtype(at::kFloat));
    at::Tensor result2 = at::zeros(grad_sample_loc_size, location.options().dtype(at::kFloat));
    at::Tensor result3 = at::zeros(grad_atten_weight_size, attn_weight.options().dtype(at::kFloat));

    at::Tensor value_fp = value.to(at::kFloat);
    at::Tensor shape_fp = shape.to(at::kInt);
    at::Tensor level_start_index_fp = level_start_index.to(at::kInt);
    at::Tensor sampling_locations_fp = location1.to(at::kFloat);
    at::Tensor attn_weight_fp = attn_weight.to(at::kFloat);
    at::Tensor grad_output_fp = grad_output.to(at::kFloat);
    EXEC_NPU_CMD(aclnnMultiScaleDeformableAttentionGrad, value_fp, shape_fp, level_start_index_fp, sampling_locations_fp,
                 attn_weight_fp, grad_output_fp, result1, result2, result3);
    result2 = result2.transpose(4, 5);
    return std::make_tuple(result1.to(ori_dtype), result2.to(ori_dtype), result3.to(ori_dtype));
}