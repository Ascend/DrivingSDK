"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-07-24
Modification Description: 
Modification 1. Add support for Ascend NPU
"""
import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class PointsInBoxAllFunction(Function):
    @staticmethod
    def forward(ctx, boxes, pts):
        result = ads_c.npu_points_in_box_all(boxes, pts)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return None

npu_points_in_box_all = PointsInBoxAllFunction.apply