"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2025-07-03
Modification Description:
Modification 1. Add support for Ascend NPU
"""

import time
import torch
import torch_npu
from torch.autograd import Function
from torch.nn import Module
from torch import Tensor
import mx_driving._C
import mx_driving


class GraphSoftmax(Function):
    @staticmethod
    def forward(ctx, src: torch.Tensor, index: torch.Tensor) -> Tensor:
        N = int(index.max()) + 1
        softmaxResult = mx_driving._C.graph_softmax(src, index.to(torch.int32), N)
        ctx.save_for_backward(src, index.to(torch.int32), softmaxResult)
        return softmaxResult

    @staticmethod
    def backward(ctx, grad_output):
        src, index, softmax_out = ctx.saved_tensors
        N = int(index.max()) + 1
        grad_output = softmax_out * grad_output
        grad_sum = mx_driving.scatter_add(grad_output, index, None, 0, N)
        grad_sum = grad_sum.index_select(0, index)
        grad_src = grad_output - softmax_out * grad_sum
        return grad_src, None


graph_softmax = GraphSoftmax.apply