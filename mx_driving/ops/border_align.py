"""
Copyright (c) OpenMMLab. All rights reserved.
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch_npu
from torch.autograd import Function

import mx_driving._C


class BorderAlignFunction(Function):
    @staticmethod
    def forward(ctx: Any, feature_map: torch.Tensor, rois: torch.Tensor, pooled_size: int) -> torch.Tensor:
        if (torch.numel(feature_map) == 0 or torch.numel(rois) == 0 or pooled_size == 0):
            raise Exception("Error! Input Tensor can not be a empty Tensor! \n")
        ctx.pooled_size = pooled_size
        ctx.feature_size = feature_map.size()
        batch_size, num_channels, data_height, data_width = feature_map.size()
        output = torch.zeros(batch_size, data_height * data_width, ctx.pooled_size + 1, num_channels).to(
            feature_map.device
        )

        mx_driving._C.border_align(feature_map, rois, output, ctx.pooled_size)

        npu_outputs, index = output.max(dim=-2)
        npu_outputs = (
            npu_outputs.reshape([batch_size, data_height * data_width, 4, num_channels // 4])
            .permute([0, 3, 1, 2])
            .contiguous()
        )
        index = (
            index.int()
            .reshape([batch_size, data_height * data_width, 4, num_channels // 4])
            .permute([0, 3, 1, 2])
            .contiguous()
        )
        ctx.save_for_backward(rois, index)

        return npu_outputs

    @staticmethod
    def backward(ctx, grad_output):
        rois, argmax_idx = ctx.saved_tensors
        _, _, height, width = ctx.feature_size
        grad_output = grad_output.contiguous()

        grad_input = mx_driving._C.border_align_backward(
            grad_output,
            rois,
            argmax_idx,
            ctx.pooled_size,
            height,
            width)
        return grad_input, None, None

border_align = BorderAlignFunction.apply
