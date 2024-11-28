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

import mx_driving._C


class FusedBiasLeakyReLU(Function):
    @staticmethod
    def forward(ctx, x, bias, negative_slope=0.2, scale=2**0.5):
        bias = torch.broadcast_to(
            bias.to(x.dtype).reshape([-1 if i == 1 else 1 for i in range(x.ndim)]), x.shape
        ).contiguous()
        out = mx_driving._C.fused_bias_leaky_relu(x, bias, negative_slope, scale)
        return out


npu_fused_bias_leaky_relu = FusedBiasLeakyReLU.apply
