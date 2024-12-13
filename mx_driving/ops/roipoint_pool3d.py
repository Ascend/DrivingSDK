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


class RoipointPool3dFunction(Function):
    @staticmethod
    def forward(ctx, num_sampled_points, points, point_features, boxes3d):
        if num_sampled_points <= 0:
            raise Exception("Input num_sampled_points be more than 0")
        if (points.size(0) != point_features.size(0)) or (points.size(0) != boxes3d.size(0)):
            raise Exception("Input points/point_features/boxes3d shape should be (B, x, x)")
        if (len(points.shape) != 3) or (points.size(2) != 3):
            raise Exception("Input points shape should be (B, N, 3)")
        if (len(point_features.shape) != 3) or (points.size(1) != point_features.size(1)):
            raise Exception("Input point_features shape should be (B, N, C)")
        if (len(boxes3d.shape) != 3) or (boxes3d.size(2) != 7):
            raise Exception("Input boxes3d shape should be (B, M, 7)")
        if (points.dtype != point_features.dtype) or (points.dtype != boxes3d.dtype):
            raise Exception("Input points/point_features/boxes3d dtype should be the same.")
        if (points.device.type != "npu") or (point_features.device.type != "npu") or (boxes3d.device.type != "npu"):
            raise ValueError("The device is not npu!")
        # points: (B, N, 3) 输入点
        # point_features: (B, N, C) 输入点特征
        # boxes3d: (B, M, 7) 边界框
        # pooled_features: (B, M, num, 3+C) 特征汇聚
        # pooled_empty_flag: (B, M) 空标志
        # float精度1/2^((126-23)/2)=3e-16 ~~ 2^127/2=1.7e34
        # half精度1/2^30=1e-9 ~~ 2^31/2=1e9
        batch_size = points.size(0)
        boxes_num = boxes3d.size(1)
        feature_len = point_features.size(2)
        pooled_features, pooled_empty_flag = mx_driving._C.npu_roipoint_pool3d_forward(
            num_sampled_points, points, point_features, boxes3d
        )
        return pooled_features, pooled_empty_flag


roipoint_pool3d = RoipointPool3dFunction.apply
