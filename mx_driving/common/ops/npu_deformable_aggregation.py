import torch
import numpy as np
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class AdsDeformableAggregation(Function):

    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(
            ctx,
            mc_ms_feat: torch.Tensor,
            spatial_shape: torch.Tensor,
            scale_start_index: torch.Tensor,
            sampling_location: torch.Tensor,
            weights: torch.Tensor):


        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        output = ads_c.npu_deformable_aggregation(
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

npu_deformable_aggregation = AdsDeformableAggregation.apply
