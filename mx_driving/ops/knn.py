"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-06-04 
Modification Description: 
Modification 1. Add support for Ascend NPU
"""

from typing import Optional

import torch
import torch_npu
from torch.autograd import Function

import mx_driving._C


class Knn(Function):
    @staticmethod
    def forward(
        ctx, k: int, xyz: torch.Tensor, center_xyz: Optional[torch.Tensor] = None, transposed: bool = False
    ) -> torch.Tensor:
        if k <= 0 and k >= 100:
            print("k should be in range (0, 100).")
            return None

        if center_xyz is None:
            center_xyz = xyz

        if transposed:
            center_xyz = center_xyz.transpose(2, 1).contiguous()
        else:
            xyz = xyz.transpose(2, 1).contiguous()

        if not xyz.is_contiguous():  # [B, 3, N]
            return None
        if not xyz.is_contiguous():  # [B, npoint, 3]
            return None

        if center_xyz.get_device() != xyz.get_device():
            print("center_xyz and xyz should be on the same device.")
            return None

        dist2, idx = mx_driving._C.knn(xyz, center_xyz, k, True)
        zeros_idx = torch.zeros(xyz.shape[0], center_xyz.shape[1], k, dtype=torch.int32).npu()
        idx.where(dist2 >= 1e10, zeros_idx)
        idx = idx.transpose(2, 1).contiguous()  # [B, k, npoint]

        return idx.int()


knn = Knn.apply
