import torch
import torch_npu
from torch.autograd import Function

import mx_driving._C


class ScatterMeanFunction(Function):
    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx, src, index, out=None, dim=0, dim_size=None):
        func = mx_driving._C.npu_scatter_mean
        res, count = func(src, index, out, dim, dim_size)
        ctx.dim = dim
        ctx.save_for_backward(index, count)
        return res

    @staticmethod
    def backward(ctx, grad_out):
        dim = ctx.dim
        index, count = ctx.saved_tensors
        result = mx_driving._C.npu_scatter_mean_grad(grad_out, index, count, dim)
        return result, None, None, None, None


scatter_mean = ScatterMeanFunction.apply
