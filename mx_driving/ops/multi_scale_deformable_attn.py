"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-06-04 
Modification Description: 
Modification 1. Add support for Ascend NPU
"""

import warnings

import torch
from torch.autograd.function import Function, once_differentiable
from torch.npu.amp import custom_bwd, custom_fwd
import mx_driving._C


class MultiScaleDeformableAttnFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(
        ctx,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
        value_level_start_index: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        value_spatial_shapes = value_spatial_shapes.int()
        value_level_start_index = value_level_start_index.int()
        sampling_locations = sampling_locations.type_as(value)
        attention_weights = attention_weights.type_as(value)

        output = mx_driving._C.multi_scale_deformable_attn(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights
        )
        ctx.save_for_backward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights
        )
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    # pylint: disable=too-many-return-values
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = mx_driving._C.multi_scale_deformable_attn_backward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output
        )
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight

    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def symbolic(
        g,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
        value_level_start_index: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
    ):
        return g.op(
            "npu::MultiScaleDeformableAttn",
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )


multi_scale_deformable_attn = MultiScaleDeformableAttnFunction.apply


def npu_multi_scale_deformable_attn_function(value, shape, offset, locations, weight):
    warnings.warn(
        "`npu_multi_scale_deformable_attn_function` will be deprecated in future. Please use `multi_scale_deformable_attn` instead.",
        DeprecationWarning,
    )
    return MultiScaleDeformableAttnFunction.apply(value, shape, offset, locations, weight)
