"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-06-04 
Modification Description: 
Modification 1. Add support for Ascend NPU
"""
import numpy as np
import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class AdsFurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, point_xyz, num_points):
        B, N = point_xyz.size()[:2]
        point_xyz = point_xyz.permute(0, 2, 1).contiguous()

        nearest_dist = torch.tensor(np.ones((B, N)) * 1e10, dtype=torch.float32, device='npu').contiguous()
        output = ads_c.npu_furthest_point_sampling(point_xyz, nearest_dist, num_points)

        return output


npu_furthest_point_sampling = AdsFurthestPointSampling.apply