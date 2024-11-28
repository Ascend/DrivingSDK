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


class ThreeInterpolateFunction(Function):

    @staticmethod
    def forward(ctx: Any, features: torch.Tensor, indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:

        b, c, m = features.size()
        n = indices.size(1)
        ctx.three_interpolate_for_backward = (indices, weight, m)

        func = mx_driving._C.npu_three_interpolate
        out = func(b, c, m, n, features, indices, weight)

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        b, c, n = grad_out.size()
        idx, weight, m = ctx.three_interpolate_for_backward

        grad_out_dtype = grad_out.dtype
        grad_out_data = grad_out.data.contiguous().to(torch.float)
        weight = weight.to(torch.float)

        grad_features = mx_driving._C.npu_three_interpolate_backward(b, c, n, m, grad_out_data, idx, weight)

        if grad_out_dtype == torch.half:
            grad_features = grad_features.to(torch.half)

        return grad_features, None, None


three_interpolate = ThreeInterpolateFunction.apply
