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

import mx_driving._C


class GridSampler2dV2(torch.autograd.Function):
    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx, input, grid, mode, padding_mode, align_corners):
        output = mx_driving._C.grid_sampler2d_v2(input, grid, mode, padding_mode, align_corners)
        return output


def grid_sampler2d_v2(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
    if (torch.numel(input) == 0 or torch.numel(grid) == 0):
        raise Exception(f"mx_driving.grid_sampler2d_v2(): Input tensor and grid tensor can not be empty tensor.\n")
    if mode != "bilinear":
        warnings.warn(
            f"mx_driving.grid_sampler2d_v2(): Not support '{mode}' mode, will call torch.nn.functional.grid_sample()."
        )
        output = torch.nn.functional.grid_sample(input, grid, mode, padding_mode, align_corners)
        return output
    if input.size(1) > 8:
        warnings.warn(
            f"mx_driving.grid_sampler2d_v2(): Not support for channel of input greater than 8, will call torch.nn.functional.grid_sample()."
        )
        output = torch.nn.functional.grid_sample(input, grid, mode, padding_mode, align_corners)
        return output
    if (padding_mode != "zeros" and padding_mode != "border" and padding_mode != "reflection"):
        raise ValueError(
            "mx_driving.grid_sampler2d_v2(): Expected padding_mode to be 'zeros', 'border', or 'reflection', "
            f"but got: '{padding_mode}'.\n"
        )

    mode_enum = 0

    if padding_mode == "zeros":
        padding_mode_enum = 0
    elif padding_mode == "border":
        padding_mode_enum = 1
    else:  # padding_mode == "reflection"
        padding_mode_enum = 2

    output = GridSampler2dV2.apply(input, grid, mode_enum, padding_mode_enum, align_corners)
    return output
