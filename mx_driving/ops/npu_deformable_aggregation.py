import numpy as np
import torch
import torch_npu
from torch.autograd import Function

import mx_driving._C


class AdsDeformableAggregation(Function):

    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(
        ctx,
        mc_ms_feat: torch.Tensor,
        spatial_shape: torch.Tensor,
        scale_start_index: torch.Tensor,
        sampling_location: torch.Tensor,
        weights: torch.Tensor,
    ):
        if (torch.numel(mc_ms_feat) == 0 or torch.numel(weights) == 0):
            raise Exception("Erorr! Input Tensor can not be a empty Tensor.\n")

        mc_ms_feat = mc_ms_feat.contiguous()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous()
        weights = weights.contiguous()

        output = mx_driving._C.npu_deformable_aggregation(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors

        if (torch.numel(mc_ms_feat) == 0 or torch.numel(spatial_shape) == 0 or torch.numel(sampling_location) == 0):
            raise Exception("Erorr! Input Tensor can not be a empty Tensor.\n")
        mc_ms_feat = mc_ms_feat.contiguous()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous()
        weights = weights.contiguous()

        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)
        grad_mc_ms_feat, grad_sampling_location, grad_weights = mx_driving._C.npu_deformable_aggregation_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            grad_output.contiguous(),
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )
        return (
            grad_mc_ms_feat,
            None,
            None,
            grad_sampling_location,
            grad_weights,
        )


npu_deformable_aggregation = AdsDeformableAggregation.apply
deformable_aggregation = AdsDeformableAggregation.apply
