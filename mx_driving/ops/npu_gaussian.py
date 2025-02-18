"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-10-06
Modification Description:
Modification 1. Add support for Ascend NPU
"""

import torch
import torch_npu
from torch.autograd import Function

import mx_driving._C


class GaussianFunction(Function):
    @staticmethod
    # pylint: disable=huawei-too-many-arguments
    def forward(
        ctx,
        boxes: torch.Tensor,
        out_size_factor,
        gaussian_overlap,
        min_radius,
        voxel_size_x,
        voxel_size_y,
        pc_range_x,
        pc_range_y,
        feature_map_size_x,
        feature_map_size_y,
        norm_bbox=True,
        with_velocity=True,
        flip_angle=False,
        max_objs=500,
    ):
        if (torch.numel(boxes) == 0):
            raise Exception("Error! Input Tensor can not be an empty Tensor.\n")
        result = mx_driving._C.npu_gaussian(
            boxes,
            out_size_factor,
            gaussian_overlap,
            min_radius,
            voxel_size_x,
            voxel_size_y,
            pc_range_x,
            pc_range_y,
            feature_map_size_x,
            feature_map_size_y,
            norm_bbox,
            with_velocity,
            flip_angle,
            max_objs
        )
        return result

npu_gaussian = GaussianFunction.apply
