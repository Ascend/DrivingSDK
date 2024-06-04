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
        # strict to fp32
        dtype_ = source.dtype
        if dtype_ == torch.float16:
            target = target.float()
            source = source.float()

        idx, dist2 = ads_c.knn(source, target, 3, False)
        if dtype_ == torch.float16:
            idx = idx.half()
            dist2 = dist2.half()
        return torch.sqrt(dist2), idx


three_nn = AdsThreeNN.apply