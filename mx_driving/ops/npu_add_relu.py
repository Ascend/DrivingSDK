"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-06-04 
Modification Description: 
Modification 1. Add support for Ascend NPU
"""

import torch
import torch.nn.functional as F
import torch_npu
from torch.autograd import Function

import mx_driving._C


class AddReluFunction(Function):
    @staticmethod
    def forward(ctx, x, y):
        if x.numel() >= 2000000:
            x = mx_driving._C.npu_add_relu(x, y)
        else:
            x = F.relu(x + y)
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        result = mx_driving._C.npu_add_relu_grad(x, grad_output)
        return result, result


npu_add_relu = AddReluFunction.apply
