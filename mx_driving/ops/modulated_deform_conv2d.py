"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-09-24
Modification Description:
Modification 1. Add support for Ascend NPU
"""

from typing import Optional, Tuple, Union

import torch
import torch_npu
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from torch.npu.amp import custom_bwd, custom_fwd
import mx_driving._C

DEVICE_NAME = torch_npu.npu.get_device_name(0)
CAST_INPUT = None if ("910_95" in DEVICE_NAME) or ("950" in DEVICE_NAME) else torch.float32

class ModulatedDeformConv2dFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=CAST_INPUT)
    # pylint: disable=huawei-too-many-arguments
    def forward(
        ctx,
        x: torch.Tensor,
        offset: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[nn.Parameter] = None,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        deformable_groups: int = 1,
    ):
        if (weight.size(2) != 3) | (weight.size(3) != 3):
            raise ValueError("Kernel size only support 3")
        if (x.size(1) % 64 != 0) | (weight.size(0) % 64 != 0):
            raise ValueError("Channel only support 64-aligned")

        ctx.kernel_size = [weight.size(2), weight.size(3)]
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups

        nhwc_x = x.permute(0, 2, 3, 1).contiguous()
        nhwc_offset = offset.permute(0, 2, 3, 1).contiguous()
        nhwc_weight = weight.permute(0, 2, 3, 1).contiguous()
        nhwc_mask = mask.permute(0, 2, 3, 1).contiguous()

        out, offset_output = mx_driving._C.modulated_deformable_conv2d(
            nhwc_x,
            nhwc_offset,
            nhwc_mask,
            nhwc_weight,
            None,
            ctx.kernel_size,
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            ctx.groups,
            ctx.deformable_groups,
            False,
        )

        _, c_in, _, _ = x.shape
        offset_output = (
            torch.tensor([]).float() if (groups == 1) and ((c_in == 256) or (c_in == 512)) else offset_output
        )
        ctx.save_for_backward(nhwc_x, nhwc_offset, nhwc_weight, nhwc_mask, offset_output)
        return out

    @staticmethod
    @once_differentiable
    @custom_bwd
    # pylint: disable=huawei-too-many-arguments,too-many-return-values
    def backward(ctx, grad_out):
        nhwc_x, nhwc_offset, nhwc_weight, nhwc_mask, offset_output = ctx.saved_tensors
        nhwc_grad_out = grad_out.permute(0, 2, 3, 1).contiguous()
        grad_x, grad_weight, _, grad_offset, grad_mask = mx_driving._C.modulated_deformable_conv2d_backward(
            nhwc_x,
            nhwc_offset,
            nhwc_mask,
            nhwc_weight,
            None,
            offset_output,
            nhwc_grad_out,
            ctx.kernel_size,
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            ctx.groups,
            ctx.deformable_groups,
            False,
        )
        grad_offset = grad_offset.reshape(grad_offset.shape[0], -1, grad_offset.shape[4], grad_offset.shape[5])
        return (
            grad_x,
            grad_offset,
            grad_mask,
            grad_weight,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def modulated_deform_conv2d(
    x: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[nn.Parameter] = None,
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
    deformable_groups: int = 1,
):
    return ModulatedDeformConv2dFunction.apply(
        x, offset, mask, weight, bias, stride, padding, dilation, groups, deformable_groups
    )
