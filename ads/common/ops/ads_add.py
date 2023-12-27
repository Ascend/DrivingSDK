import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class AdsAddFunction(Function):
    @staticmethod
    def forward(ctx, input1, input2):
        result = ads_c.npu_ads_add(input1, input2)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return None

npu_ads_add = AdsAddFunction.apply