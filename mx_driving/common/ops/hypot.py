"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""

import torch
import torch_npu
from torch.autograd import Function

import mx_driving._C


class Hypot(Function):
    @staticmethod
    def forward(ctx, x, y):
        x_broadcasted, y_broadcasted = torch.broadcast_tensors(x, y)
        out = mx_driving._C.npu_hypot(x_broadcasted.contiguous(), y_broadcasted.contiguous())
        ctx.save_for_backward(x, y, out)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        x, y, out = ctx.saved_tensors
        x_broadcasted, y_broadcasted = torch.broadcast_tensors(x, y)
        x_grad, y_grad = mx_driving._C.npu_hypot_grad(
            x_broadcasted.contiguous(), y_broadcasted.contiguous(), out, out_grad
        )

        # reshape the broadcasted tensors to origin tensors and sum the grad
        for dim, size in enumerate(x.shape):
            if size == 1:
                x_grad = x_grad.sum(dim, keepdim=True)
        for dim, size in enumerate(y.shape):
            if size == 1:
                y_grad = y_grad.sum(dim, keepdim=True)

        return x_grad, y_grad


hypot = Hypot.apply
