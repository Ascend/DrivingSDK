from typing import Optional
import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class AdsKnn(Function):
    @staticmethod
    def forward(ctx,
                k: int,
                xyz: torch.Tensor,
                center_xyz: Optional[torch.Tensor] = None,
                transposed: bool = False) -> torch.Tensor:
        if k <= 0 and k >= 100:
            print('k should be in range (0, 100).')
            return None

        if center_xyz is None:
            center_xyz = xyz

        if not transposed:
            xyz = xyz.transpose(1, 2).contiguous()
        else:
            center_xyz = center_xyz.transpose(1, 2).contiguous()

        if not xyz.is_contiguous(): # [B, 3, N]
            return None
        if not xyz.is_contiguous(): # [B, npoint, 3]
            return None

        if center_xyz.get_device() != xyz.get_device():
            print('center_xyz and xyz should be on the same device.')
            return None

        idx, dist2 = ads_c.knn(xyz, center_xyz, k, True)
        idx = idx.transpose(1, 2).contiguous() # [B, k, npoint]

        return idx


knn = AdsKnn.apply