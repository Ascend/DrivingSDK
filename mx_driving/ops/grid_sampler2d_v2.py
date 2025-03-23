"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
Modification by: Huawei Developers
Modification date: 2025-01-07
Modification Description:
Modification 1. Add support for Ascend NPU
"""

import warnings
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch

import mx_driving._C


class GridSampler2dV2Function(Function):
    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx, input_tensor, grid_tensor, mode='bilinear', padding_mode='zeros', align_corners=False):
        if (torch.numel(input_tensor) == 0 or torch.numel(grid_tensor) == 0):
            raise Exception(f"mx_driving.grid_sampler2d_v2(): Input tensor and grid tensor can not be empty tensor.\n")
        if input_tensor.size(1) > 128:
            warnings.warn(
                f"mx_driving.grid_sampler2d_v2(): Not support for channel of input greater than 128, will call torch.nn.functional.grid_sample()."
            )
            output = torch.nn.functional.grid_sample(input_tensor, grid_tensor, mode, padding_mode, align_corners)
            return output
        if mode != "bilinear":
            warnings.warn(
                f"mx_driving.grid_sampler2d_v2(): Not support '{mode}' mode, will call torch.nn.functional.grid_sample()."
            )
            output = torch.nn.functional.grid_sample(input_tensor, grid_tensor, mode, padding_mode, align_corners)
            return output
        if (
                padding_mode != "zeros"
                and padding_mode != "border"
        ):
            raise ValueError(
                "nn.functional.grid_sample(): expected padding_mode to be 'zeros', 'border', "
                f"but got: '{padding_mode}'"
            )
        ctx.interpolation = mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        interpolation_mode_map = {'bilinear': 0, 'nearest': 1}
        interpolation = interpolation_mode_map.get(mode, 0)
        padding_mode_map = {'zeros': 0, 'border': 1, 'reflection': 2}
        padding = padding_mode_map.get(padding_mode, 0)
        output = mx_driving._C.grid_sampler2d_v2(input_tensor, grid_tensor, interpolation, padding, align_corners)
        ctx.save_for_backward(input_tensor, grid_tensor)
        return output

    @staticmethod
    @once_differentiable
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def backward(ctx, grad_output):
        input_x, input_grid = ctx.saved_tensors
        interpolation_mode = ctx.interpolation
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners
        nhwc_input_x = input_x.permute(0, 2, 3, 1).contiguous()
        nhwc_grad_output = grad_output.permute(0, 2, 3, 1).contiguous()
        interpolation_mode_map = {'bilinear': 0, 'nearest': 1}
        interpolation = interpolation_mode_map.get(interpolation_mode, 0)
        padding_mode_map = {'zeros': 0, 'border': 1, 'reflection': 2}
        padding = padding_mode_map.get(padding_mode, 0)
        grad_x, grad_grid = mx_driving._C.grid_sampler2d_v2_backward(nhwc_grad_output, nhwc_input_x, input_grid,
                                                                     interpolation, padding, align_corners)
        return grad_x, grad_grid, None, None, None

grid_sampler2d_v2 = GridSampler2dV2Function.apply
