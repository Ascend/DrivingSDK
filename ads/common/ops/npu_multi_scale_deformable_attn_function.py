import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class MultiScaleDeformableAttnFunction(Function):
    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx, value, shape, offset, locations, weight):
        value_trans = torch.permute(value, (0, 2, 1, 3)).contiguous()
        locations_trans = torch.permute(locations, (0, 2, 3, 5, 1, 4)).contiguous()
        weight_trans = torch.permute(weight, (0, 2, 3, 1, 4)).contiguous()
        result = ads_c.npu_multi_scale_deformable_attn_function(value_trans, shape, offset, locations_trans, weight_trans)
        ctx.save_for_backward(value, shape, offset, locations, weight)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        value, shape, offset, locations, weight = ctx.saved_tensors
        locations_trans = torch.permute(locations, (0, 2, 3, 4, 5, 1)).contiguous()
        weight_trans = torch.permute(weight, (0, 2, 3, 4, 1)).contiguous()
        grad_value, grad_locations_trans, grad_weight_trans = ads_c.multi_scale_deformable_attn_grad_v2(value, shape, offset,
                                                                                                            locations_trans, weight_trans,
                                                                                                            grad_output)
        grad_locations = torch.permute(grad_locations_trans, (0, 5, 1, 2, 3, 4)).contiguous()
        grad_weight = torch.permute(grad_weight_trans, (0, 4, 1, 2, 3)).contiguous()
        return grad_value, None, None, grad_locations, grad_weight


npu_multi_scale_deformable_attn_function = MultiScaleDeformableAttnFunction.apply
