"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-06-04 
Modification Description: 
Modification 1. Add support for Ascend NPU
"""

import warnings

import torch
import torch_npu
from torch.autograd import Function

import mx_driving._C


class PointsInBoxFunction(Function):
    @staticmethod
    def forward(ctx, boxes, pts):
        result = mx_driving._C.npu_points_in_box(boxes, pts)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return None


points_in_box = PointsInBoxFunction.apply


def npu_points_in_box(boxes, pt):
    warnings.warn(
        "`npu_points_in_box` will be deprecated in future. Please use `points_in_box` instead.",
        DeprecationWarning,
    )
    return PointsInBoxFunction.apply(boxes, pt)