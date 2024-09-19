"""
Copyright (c) OpenMMLab. All rights reserved.
"""
from typing import Any, Optional, Tuple, Union

import torch
import torch_npu
import torch.nn as nn
from torch.autograd import Function

import ads_c


class RoIAlignRotatedV2Function(Function):
    @staticmethod
    def forward(ctx: Any, feature_map: torch.Tensor, rois: torch.Tensor, spatial_scale: float,
                sampling_ratio: int, pooled_height: int, pooled_width: int, aligned: bool = True, clockwise: bool = False) -> torch.Tensor:
        ctx.pooled_height = pooled_height
        ctx.pooled_width = pooled_width
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.aligned = aligned
        ctx.clockwise = clockwise
        ctx.save_for_backward(feature_map, rois)
        ctx.feature_size = feature_map.size()
        batch_size, num_channels, data_height, data_width = feature_map.size()
        num_rois = rois.size(0)

        output = feature_map.new_zeros(num_rois, ctx.pooled_height, ctx.pooled_width, num_channels).to(feature_map.device)

        ads_c.roi_align_rotated_v2_forward_npu(
            feature_map,
            rois,
            output,
            ctx.spatial_scale,
            ctx.sampling_ratio,
            ctx.pooled_height,
            ctx.pooled_width,
            ctx.aligned,
            ctx.clockwise)
        output = output.transpose(2, 3).transpose(1, 2).contiguous()
        return output
    
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        feature_map, rois = ctx.saved_tensors
        rois_trans = torch.permute(rois, (1, 0)).contiguous()
        grad_output_trans = torch.permute(grad_output, (0, 2, 3, 1)).contiguous()
        grad_feature_map = ads_c.npu_roi_align_rotated_grad_v2(
            feature_map, rois_trans, grad_output_trans,
            ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
            ctx.sampling_ratio, ctx.aligned, ctx.clockwise)
        grad_feature_map = grad_feature_map.permute(0, 3, 1, 2).contiguous()
        
        return grad_feature_map, None, None, None, None, None, None, None

roi_align_rotated_v2 = RoIAlignRotatedV2Function.apply