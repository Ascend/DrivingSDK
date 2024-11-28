"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-06-04 
Modification Description: 
Modification 1. Add support for Ascend NPU
"""

from typing import Any, Tuple

import torch
import torch_npu
from torch.autograd import Function

import mx_driving._C


class ThreeNN(Function):
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

        dist2, idx = mx_driving._C.knn(source, target, 3, False)
        dist2 = torch.sqrt(dist2)
        if dtype_ == torch.float16:
            dist2 = dist2.half()
        return dist2, idx.int()


three_nn = ThreeNN.apply
