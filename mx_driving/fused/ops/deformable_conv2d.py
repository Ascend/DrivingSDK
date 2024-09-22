"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-06-04
Modification Description:
Modification 1. Add support for Ascend NPU
"""

import torch
from torch.autograd import Function
import torch_npu
import ads_c


class DeformableConv2dFunction(Function):
    @staticmethod
    def forward(ctx,
                x: torch.Tensor,
                weight: torch.Tensor,
                offset: torch.Tensor,
                bias: torch.Tensor,
                kernel_size,
                stride, padding,
                dilation, groups=1,
                deformable_groups=1,
                modulated=False,
                xoffsets_transpose=True):
        out, x_offset = ads_c.npu_deformable_conv2d(x, offset, weight, bias, kernel_size, stride, padding, dilation, groups, deformable_groups, modulated, xoffsets_transpose)
        return out, x_offset
    
deformable_conv2d = DeformableConv2dFunction.apply