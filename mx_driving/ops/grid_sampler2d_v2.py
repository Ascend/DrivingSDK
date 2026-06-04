"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
Modification by: Huawei Developers
Modification date: 2025-01-07
Modification Description:
Modification 1. Add support for Ascend NPU
"""

import warnings
import torch
import torch_npu
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import mx_driving._C


class GridSampler2dV2Function(Function):
    @staticmethod
    def forward(ctx, input_tensor, grid_tensor, mode="bilinear", padding_mode="zeros", align_corners=False):
        if torch.numel(input_tensor) == 0 or torch.numel(grid_tensor) == 0:
            raise ValueError("mx_driving.grid_sampler2d_v2(): Input tensor and grid tensor can not be empty tensor.\n")
        if input_tensor.size(1) > 128:
            warnings.warn(
                "mx_driving.grid_sampler2d_v2(): Not support for channel of input greater than 128, will call torch.nn.functional.grid_sample()."
            )
            output = torch.nn.functional.grid_sample(input_tensor, grid_tensor, mode, padding_mode, align_corners)
            return output
        if mode != "bilinear":
            warnings.warn(
                f"mx_driving.grid_sampler2d_v2(): Not support '{mode}' mode, will call torch.nn.functional.grid_sample()."
            )
            output = torch.nn.functional.grid_sample(input_tensor, grid_tensor, mode, padding_mode, align_corners)
            return output
        if padding_mode not in ('zeros', 'border'):
            raise ValueError(
                f"nn.functional.grid_sample(): expected padding_mode to be 'zeros', 'border', but got: '{padding_mode}'"
            )
        ctx.interpolation = mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        interpolation_mode_map = {"bilinear": 0, "nearest": 1}
        interpolation = interpolation_mode_map.get(mode, 0)
        padding_mode_map = {"zeros": 0, "border": 1, "reflection": 2}
        padding = padding_mode_map.get(padding_mode, 0)
        output = mx_driving._C.grid_sampler2d_v2(input_tensor, grid_tensor, interpolation, padding, align_corners)
        ctx.save_for_backward(input_tensor, grid_tensor)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input_x, input_grid = ctx.saved_tensors
        interpolation_mode = ctx.interpolation
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners
        nhwc_input_x = input_x.permute(0, 2, 3, 1).contiguous()
        nhwc_grad_output = grad_output.permute(0, 2, 3, 1).contiguous()
        interpolation_mode_map = {"bilinear": 0, "nearest": 1}
        interpolation = interpolation_mode_map.get(interpolation_mode, 0)
        padding_mode_map = {"zeros": 0, "border": 1, "reflection": 2}
        padding = padding_mode_map.get(padding_mode, 0)
        grad_x, grad_grid = mx_driving._C.grid_sampler2d_v2_backward(  # pylint: disable=unpacking-non-sequence
            nhwc_grad_output, nhwc_input_x, input_grid, interpolation, padding, align_corners
        )
        return grad_x, grad_grid, None, None, None


def grid_sampler2d_v2(input_tensor, grid_tensor, mode="bilinear", padding_mode="zeros", align_corners=False):
    DEVICE_NAME = torch_npu.npu.get_device_name(input_tensor.device.index)
    if "Ascend910" in DEVICE_NAME:
        return GridSampler2dV2Function.apply(input_tensor, grid_tensor, mode, padding_mode, align_corners)
    elif "Ascend950" in DEVICE_NAME:
        return torch.nn.functional.grid_sample(
            input_tensor, grid_tensor, mode=mode, padding_mode=padding_mode, align_corners=align_corners
        )
    else:
        raise NotImplementedError("The grid_sampler2d_v2 currently only supports Ascend910B, Ascend910C and Ascend950.")
