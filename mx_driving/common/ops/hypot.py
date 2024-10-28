"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""
import torch
from torch.autograd import Function

import torch_npu
import ads_c


class Hypot(Function):
    @staticmethod
    def forward(ctx, x, y):
        x_broadcasted, y_broadcasted = torch.broadcast_tensors(x, y)
        out = ads_c.npu_hypot(x_broadcasted.contiguous(), y_broadcasted.contiguous())
        ctx.save_for_backward(x, y, out);
        return out

    @staticmethod
    def backward(ctx, out_grad):
        x, y, out = ctx.saved_tensors
        x_broadcasted, y_broadcasted = torch.broadcast_tensors(x, y)
        x_grad, y_grad = ads_c.npu_hypot_grad(x_broadcasted.contiguous(), y_broadcasted.contiguous(), out, out_grad)

        # reshape the broadcasted tensors to origin tensors and sum the grad
        for dim, size in enumerate(x.shape):
            if size == 1:
                x_grad = x_grad.sum(dim, keepdim=True)
        for dim, size in enumerate(y.shape):
            if size == 1:
                y_grad = y_grad.sum(dim, keepdim=True)

        return x_grad, y_grad

hypot = Hypot.apply
