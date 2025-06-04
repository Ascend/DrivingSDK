"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
Modification by: Huawei Developers
Modification date: 2025-05-30
Modification Description:
Modification 1. Add support for Ascend NPU
"""

from typing import Any, Tuple

import torch
from torch.autograd import Function
from torch.nn import Module

import mx_driving._C


class GridSamplerFunction(Function):

    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx, features: torch.Tensor, grid: torch.Tensor, interpolation='bilinear', padding='zeros', align=True):
        
        out = torch.nn.functional.grid_sample(features, grid, interpolation, padding, align)
        ctx.save_for_backward(features, grid)
        ctx.interpolation = interpolation
        ctx.padding = padding
        ctx.align = align

        return out

    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def backward(ctx, grad: torch.Tensor):
        
        x, grid = ctx.saved_tensors
        interpolation, padding, align = ctx.interpolation, ctx.padding, ctx.align

        interpolation_mode_map = {'bilinear': 0, 'nearest': 1}
        interpolation_mode = interpolation_mode_map.get(interpolation, 0)
        padding_mode_map = {'zeros': 0, 'border': 1, 'reflection': 2}
        padding_mode = padding_mode_map.get(padding, 0)

        dx, dgrid = mx_driving._C.grid_sampler3d_grad_v1(grad, x, grid, interpolation_mode, padding_mode, align)
        
        return dx, dgrid, None, None, None

grid_sampler3d_v1 = GridSamplerFunction.apply
