import torch
from torch.autograd import Function

import torch_npu
import mx_driving._C


class GeometricKernelAttentionFunc(Function):
    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx, value, spatial_shapes, level_start_index, sampling_locations, attn_weights):
        result = mx_driving._C.npu_geometric_kernel_attention_func(
            value, spatial_shapes, level_start_index, sampling_locations, attn_weights
        )
        ctx.save_for_backward(value, spatial_shapes, level_start_index, sampling_locations, attn_weights)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        value, spatial_shapes, level_start_index, sampling_locations, attn_weights = ctx.saved_tensors
        grad_value, grad_attn_weights = mx_driving._C.npu_geometric_kernel_attention_backward(
            value, spatial_shapes, level_start_index, sampling_locations, attn_weights, grad_output
        )
        return grad_value, None, None, None, grad_attn_weights


npu_geometric_kernel_attention_func = GeometricKernelAttentionFunc.apply
