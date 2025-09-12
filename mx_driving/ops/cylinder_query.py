"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2025-09-10
Modification Description:
Modification 1. Add support for Ascend NPU
"""

import torch
from torch.autograd import Function
import torch_npu
import mx_driving._C


class CylinderQuery(Function):
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    @staticmethod
    def forward(ctx, radius, hmin, hmax, nsample, new_xyz, xyz, rot):
        rot = rot.reshape(rot.shape[0], rot.shape[1], 9)
        group_idx = mx_driving._C.cylinder_query(radius, hmin, hmax, nsample, new_xyz, xyz, rot)
        out = CylinderQuery.sortRes(group_idx, nsample)
        return out
    
    @staticmethod
    def backward(ctx, gradout):
        return ()
    
    @classmethod
    def sortRes(cls, group_idx, nsample):
        b = group_idx.shape[0]
        m = group_idx.shape[1]
        n = group_idx.shape[2]
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        mask = group_idx >= n
        group_first = torch.where(mask, 0, group_idx)
        group_idx = torch.where(mask, group_first[..., 0:1], group_idx)
        return group_idx.to(dtype=torch.int32)
    
    
cylinder_query = CylinderQuery.apply