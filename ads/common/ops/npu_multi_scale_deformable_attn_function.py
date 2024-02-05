import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class MultiScaleDeformableAttnFunction(Function):
    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx, value, shape, offset, locations, weight):
        result = ads_c.npu_multi_scale_deformable_attn_function(value, shape, offset, locations, weight)
        ctx.save_for_backward(value, shape, offset, locations, weight)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        value, shape, offset, locations, weight = ctx.saved_tensors
        grad_value, grad_sample_loc, grad_atten_weight = ads_c.multi_scale_deformable_attn_grad(value, shape, offset,
                                                                                                locations, weight,
                                                                                                grad_output.contiguous())
        return grad_value, None, None, grad_sample_loc, grad_atten_weight


npu_multi_scale_deformable_attn_function = MultiScaleDeformableAttnFunction.apply
