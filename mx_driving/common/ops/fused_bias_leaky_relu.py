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
from torch.nn import Module

import torch_npu
import ads_c


class FusedBiasLeakyReluFunction(Function):
    @staticmethod
    def forward(ctx, x, bias, negative_slop=0.2, scale=2**0.5):
        y = ads_c.fused_bias_leaky_relu(x, bias, negative_slop, scale)
        return y

npu_fused_bias_leaky_relu = FusedBiasLeakyReluFunction.apply