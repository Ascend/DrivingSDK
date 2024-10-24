"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""
import torch
from torch.autograd import Function

import torch_npu
import ads_c


class Hypot(Function):
    @staticmethod
    def forward(ctx, x, y):
        x_broadcasted, y_broadcasted = torch.broadcast_tensors(x, y)
        out = ads_c.npu_hypot(x_broadcasted.contiguous(), y_broadcasted.contiguous())
        return out

hypot = Hypot.apply
