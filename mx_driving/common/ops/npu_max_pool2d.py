"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-06-04 
Modification Description: 
Modification 1. Add support for Ascend NPU
"""
import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class MaxPool2d(Function):
    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx, x, kernel_size, stride, padding):
        x_trans = torch.permute(x, (0, 2, 3, 1)).contiguous()
        y_trans = ads_c.npu_max_pool2d(x_trans, kernel_size, stride, padding)
        y = torch.permute(y_trans, (0, 3, 1, 2)).contiguous()
        return y

npu_max_pool2d = MaxPool2d.apply
