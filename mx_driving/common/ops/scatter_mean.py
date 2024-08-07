import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class ScatterMeanFunction(Function):
    @staticmethod
    def forward(ctx, src, index, out=None, dim=0, dim_size=None):
        func = ads_c.npu_scatter_mean
        res, count = func(src, index, out, dim, dim_size)
        return res

scatter_mean = ScatterMeanFunction.apply
