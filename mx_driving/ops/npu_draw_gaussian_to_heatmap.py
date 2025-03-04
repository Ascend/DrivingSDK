"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
Modification by: Huawei Developers
Modification date: 2025-03-04
Modification Description:
Modification 1. Add support for Ascend NPU
"""

import torch
import torch_npu
from torch.autograd import Function

import mx_driving._C


class DrawGaussianToHeatmapFunction(Function):
    @staticmethod
    # pylint: disable=huawei-too-many-arguments
    def forward(
        ctx,
        mask,
        cur_class_id,
        center_int,
        radius,
        feature_map_size_x,
        feature_map_size_y,
        num_classes
    ):
        heatmap = mx_driving._C.npu_draw_gaussian_to_heatmap(
            mask,
            cur_class_id,
            center_int,
            radius,
            feature_map_size_x,
            feature_map_size_y,
            num_classes
        )
        return heatmap


npu_draw_gaussian_to_heatmap = DrawGaussianToHeatmapFunction.apply
