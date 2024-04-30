import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class ScatterMeanGradFunction(Function):
    @staticmethod
    def forward(ctx, grad_out, index, dim):
        result = ads_c.npu_scatter_mean_grad(grad_out, index, dim)
        return result
npu_scatter_mean_grad = ScatterMeanGradFunction.apply