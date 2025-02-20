"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""

import torch
import torch.nn.functional as F
import torch_npu
from torch.autograd import Function

import mx_driving._C


class BacthMatmulFunction(Function):
    @staticmethod
    def forward(ctx, projection_mat, pts_extend):
        broadcast_shape = [max(a, b) for a, b in zip(projection_mat.shape, pts_extend.shape)]
        projection_mat = projection_mat.expand(broadcast_shape).contiguous()
        pts_extend = pts_extend.expand(broadcast_shape).contiguous()
        result = mx_driving._C.npu_batch_matmul(projection_mat, pts_extend)
        result = result.sum(dim=-1, keepdim=True)
        ctx.save_for_backward(projection_mat, pts_extend)
        return result

    @staticmethod
    def backward(ctx, grad):
        (projection_mat, pts_extend) = ctx.saved_tensors
        broadcast_shape = projection_mat.shape
        grad = grad.expand(broadcast_shape).contiguous()
        dx = mx_driving._C.npu_batch_matmul(grad, pts_extend)
        dw = mx_driving._C.npu_batch_matmul(projection_mat, grad)
        dw = dw.sum(dim=-2, keepdim=True)
        return dx, dw


npu_batch_matmul = BacthMatmulFunction.apply
