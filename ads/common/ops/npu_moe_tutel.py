import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class MoeTutelFunction(Function):
    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx, x, gates, indices, locations, capacity):
        result = ads_c.npu_moe_tutel(x, gates, indices, locations, capacity)
        ctx.save_for_backward(x, gates, indices, locations)
        return result

    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    # 'pylint: disable=too-many-return-arguments,huawei-too-many-return-arguments
    def backward(ctx, y_grad):
        x0, gates, indices, locations = ctx.saved_tensors
        x_grad = ads_c.npu_moe_tutel_data_backward(y_grad, gates, indices, locations)
        gates_grad = ads_c.npu_moe_tutel_gate_backward(x0, y_grad, indices, locations)
        return x_grad, gates_grad, None, None, None


npu_moe_tutel = MoeTutelFunction.apply
