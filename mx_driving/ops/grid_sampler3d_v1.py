"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
Modification by: Huawei Developers
Modification date: 2025-05-30
Modification Description:
Modification 1. Add support for Ascend NPU
"""

import torch
import torch_npu
from torch.autograd import Function

import mx_driving._C


class GridSamplerFunction(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, grid: torch.Tensor, interpolation="bilinear", padding="zeros", align=True):
        out = torch.nn.functional.grid_sample(features, grid, interpolation, padding, align)
        ctx.save_for_backward(features, grid)
        ctx.interpolation = interpolation
        ctx.padding = padding
        ctx.align = align

        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        x, grid = ctx.saved_tensors
        interpolation, padding, align = ctx.interpolation, ctx.padding, ctx.align

        interpolation_mode_map = {"bilinear": 0, "nearest": 1}
        interpolation_mode = interpolation_mode_map.get(interpolation, 0)
        padding_mode_map = {"zeros": 0, "border": 1, "reflection": 2}
        padding_mode = padding_mode_map.get(padding, 0)

        dx, dgrid = mx_driving._C.grid_sampler3d_grad_v1(  # pylint: disable=unpacking-non-sequence
            grad, x, grid, interpolation_mode, padding_mode, align
        )

        return dx, dgrid, None, None, None


def grid_sampler3d_v1(
    features: torch.Tensor, grid: torch.Tensor, interpolation="bilinear", padding="zeros", align=True
):
    DEVICE_NAME = torch_npu.npu.get_device_name(features.device.index)
    if "Ascend910" in DEVICE_NAME:
        return GridSamplerFunction.apply(features, grid, interpolation, padding, align)
    elif "Ascend950" in DEVICE_NAME:
        return torch.nn.functional.grid_sample(
            features, grid, mode=interpolation, padding_mode=padding, align_corners=align
        )
    else:
        raise NotImplementedError("The grid_sampler3d_v1 currently only supports Ascend910B, Ascend910C and Ascend950.")
