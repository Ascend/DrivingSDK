"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-10-16
Modification Description: 
Modification 1. Add support for Ascend NPU
"""

from typing import Any, Tuple, Union

import torch
import torch_npu
from torch.autograd import Function

import mx_driving._C


def is_tuple_of(input_tuple, expected_type=int):
    for item in input_tuple:
        if not isinstance(item, expected_type):
            return False
    return True


class RoIAwarePool3dFunction(Function):
    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(
        ctx: Any,
        rois: torch.Tensor,
        pts: torch.Tensor,
        pts_feature: torch.Tensor,
        out_size: Union[int, tuple],
        max_pts_per_voxel: int,
        mode: int,
    ):
        if (out_size == 0):
            raise Exception("Error! out_size can not be 0.\n")
        
        if (max_pts_per_voxel == 0):
            raise Exception("Error! max_pts_per_voxel can not be 0.\n")
        
        if isinstance(out_size, int):
            out_x = out_y = out_z = out_size
        elif len(out_size) == 3 or is_tuple_of(out_size, int):
            out_x, out_y, out_z = out_size
        else:
            raise Exception("outsize attr Error!\n")

        num_rois = rois.shape[0]
        num_channels = pts_feature.shape[-1]
        num_pts = pts.shape[0]

        pooled_features = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels))
        argmax = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels), dtype=torch.int32)
        pts_idx_of_voxels = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, max_pts_per_voxel), dtype=torch.int32)

        mx_driving._C.npu_roiaware_pool3d_forward(
            rois, pts, pts_feature, argmax, pts_idx_of_voxels, pooled_features, mode
        )

        ctx.save_for_backward(pts_idx_of_voxels, argmax, mode, num_pts, num_channels)

        return pooled_features

    @staticmethod
    # pylint: disable=too-many-return-values
    def backward(ctx: Any, grad_out: torch.Tensor):
        if (torch.numel(grad_out) == 0):
            raise Exception("Error! Input Tensor can not be a empty Tensor.\n")
        
        ret = ctx.roiaware_pool3d_for_backward
        pts_idx_of_voxels, argmax, mode, num_pts, num_channels = ret

        # backward
        grad_in = mx_driving._C.roiaware_pool3d_grad(
            pts_idx_of_voxels, argmax, grad_out.contiguous(), num_pts, pool_method=mode
        )

        return None, None, grad_in, None, None, None


roiaware_pool3d = RoIAwarePool3dFunction.apply
