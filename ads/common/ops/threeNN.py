from typing import Any, Tuple
import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class AdsThreeNN(Function):
    @staticmethod
    def forward(ctx: Any, target: torch.Tensor, source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # target is center_xyz
        target = target.contiguous()
        source = source.transpose(1, 2).contiguous()

        idx, dist2 = ads_c.knn(source, target, 3, False)
        return torch.sqrt(dist2), idx


three_nn = AdsThreeNN.apply