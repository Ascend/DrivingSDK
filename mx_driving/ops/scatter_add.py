import torch
from torch.autograd import Function
import mx_driving._C


class ScatterAddFunction(Function):
    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx, src, index, out=None, dim=0, dim_size=None):
        src_dtype = src.dtype
        res = mx_driving._C.npu_scatter_add(src.float(), index, out, dim, dim_size).to(src_dtype)
        ctx.dim = dim
        ctx.save_for_backward(index)
        return res

    @staticmethod
    def backward(ctx, grad_out):
        dim = ctx.dim
        index, = ctx.saved_tensors
        grad_out_dtype = grad_out.dtype
        result = mx_driving._C.npu_scatter_add_grad(grad_out.float(), index, dim).to(grad_out_dtype)
        return result, None, None, None, None

scatter_add = ScatterAddFunction.apply