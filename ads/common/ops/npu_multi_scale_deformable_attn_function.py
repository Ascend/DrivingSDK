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
        return result

npu_multi_scale_deformable_attn_function = MultiScaleDeformableAttnFunction.apply
