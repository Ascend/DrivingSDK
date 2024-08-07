"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-06-04 
Modification Description: 
Modification 1. Add support for Ascend NPU
"""
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
        locations_trans = torch.permute(locations, (0, 1, 5, 3, 2, 4)).contiguous()
        weight_trans = torch.permute(weight, (0, 1, 3, 2, 4)).contiguous()
        result = ads_c.npu_multi_scale_deformable_attn_function(value_trans, shape, offset, locations_trans,
                                                                weight_trans)
        ctx.save_for_backward(value_trans, shape, offset, locations_trans, weight_trans)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        value_trans, shape, offset, locations_trans, weight_trans = ctx.saved_tensors
        grad_value_trans, grad_locations_trans, grad_weight = ads_c.multi_scale_deformable_attn_grad(value_trans, shape,
                                                                                                     offset,
                                                                                                     locations_trans,
                                                                                                     weight_trans,
                                                                                                     grad_output)
        grad_value = torch.permute(grad_value_trans, (0, 2, 1, 3)).contiguous()
        grad_locations = torch.permute(grad_locations_trans, (0, 1, 2, 3, 5, 4)).contiguous()
        return grad_value, None, None, grad_locations, grad_weight


npu_multi_scale_deformable_attn_function = MultiScaleDeformableAttnFunction.apply
