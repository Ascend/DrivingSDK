"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-06-04 
Modification Description: 
Modification 1. Add support for Ascend NPU
"""

import torch
import torch_npu
from torch.autograd import Function

import mx_driving._C


class ScatterMaxFunction(Function):
    @staticmethod
    def forward(ctx, updates, indices, out=None):
        func = mx_driving._C.scatter_max_with_argmax_v2
        out, argmax = func(updates, indices, out)
        ctx.save_for_backward(argmax, updates)
        return out, argmax

    @staticmethod
    def backward(ctx, grad_output, grad_argmax):
        argmax, updates = ctx.saved_tensors

        device = argmax.device
        grad_updates_index0 = argmax.unsqueeze(-1)
        grad_updates_index1 = (
            torch.tile(torch.arange(0, argmax.shape[1]), argmax.shape[0:1:1])
            .reshape(argmax.shape)
            .unsqueeze(-1)
            .to(device)
        )
        grad_updates_indices = torch.concat((grad_updates_index0, grad_updates_index1), -1).to(device)
        grad_updates_indices_uss = (
            grad_updates_indices[..., 0] * grad_updates_indices.shape[1] + grad_updates_indices[..., 1]
        )
        num_segments = torch.tensor(updates.shape[0] * updates.shape[1]).to(device)

        grad = mx_driving._C.npu_scatter_max_backward(grad_output, grad_updates_indices_uss, num_segments)

        return grad.reshape(updates.shape), None, None


scatter_max = ScatterMaxFunction.apply
