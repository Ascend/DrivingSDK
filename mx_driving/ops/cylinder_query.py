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
    
    @classmethod
    def sortRes(cls, group_idx, nsample):
        b = group_idx.shape[0]
        m = group_idx.shape[1]
        n = group_idx.shape[2]
        sorted_idx = group_idx.sort(dim=-1)[0]
        head, _ = torch.split(sorted_idx, [nsample, sorted_idx.size(-1) - nsample], dim=-1)
        group_idx = head
        mask = group_idx >= n
        group_first = (~mask).float() * group_idx
        group_idx = torch.lerp(group_idx, group_first[..., 0:1], mask.float())

        return group_idx.to(dtype=torch.int32)
    
    
cylinder_query = CylinderQuery.apply