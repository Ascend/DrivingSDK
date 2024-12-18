"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-06-04 
Modification Description: 
Modification 1. Add support for Ascend NPU
"""
import warnings

import numpy as np
import torch
import torch_npu
from torch.autograd import Function

import mx_driving._C


class AdsFurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, point_xyz, num_points):
        if (torch.numel(point_xyz) == 0):
            raise Exception("Error! Input Tensor can not be a empty Tensor.\n")
        
        if (num_points == 0):
            raise Exception("Error! num_points can not zero.\n")
        
        B, N = point_xyz.size()[:2]
        point_xyz = point_xyz.permute(0, 2, 1).contiguous()

        nearest_dist = torch.tensor(np.ones((B, N)) * 1e10, dtype=torch.float32, device="npu").contiguous()
        output = mx_driving._C.npu_furthest_point_sampling(point_xyz, nearest_dist, num_points)

        return output


def furthest_point_sampling(point_xyz, num_points):
    return AdsFurthestPointSampling.apply(point_xyz, num_points)


def npu_furthest_point_sampling(point_xyz, num_points):
    warnings.warn(
        "`npu_furthest_point_sampling` will be deprecated in future. Please use `furthest_point_sampling` instead.",
        DeprecationWarning,
    )
    return AdsFurthestPointSampling.apply(point_xyz, num_points)
