"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-06-04 
Modification Description: 
Modification 1. Add support fro Ascend NPU
"""
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
        source = source.transpose(2, 1).contiguous()
        # strict to fp32
        dtype_ = source.dtype
        if dtype_ == torch.float16:
            target = target.float()
            source = source.float()

        dist = ads_c.knn(source, target, 3, False)
        dist2, idx = torch.topk(dist, 3, dim=2, largest=False, sorted=True)
        dist2 = torch.sqrt(dist2)
        if dtype_ == torch.float16:
            dist2 = dist2.half()
        return dist2, idx.type(torch.IntTensor)


three_nn = AdsThreeNN.apply