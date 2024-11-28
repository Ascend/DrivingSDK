"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-06-04 
Modification Description: 
Modification 1. Add support for Ascend NPU
"""

import torch
import torch_npu
from torch.autograd import Function

import mx_driving._C


class AdsFurthestPointSamplingWithDistFunction(Function):
    @staticmethod
    def forward(ctx, points_dist, num_points):
        B, N = points_dist.size()[:2]
        nearest_temp = points_dist.new_zeros([B, N]).fill_(1e10)
        result = mx_driving._C.furthest_point_sampling_with_dist(points_dist, nearest_temp, num_points)
        return result


furthest_point_sample_with_dist = AdsFurthestPointSamplingWithDistFunction.apply
