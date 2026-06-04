"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-06-04
Modification Description:
Modification 1. Add support for Ascend NPU
"""

import torch
import torch_npu
from torch.autograd import Function
import mx_driving._C


class MaxPool2d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride, padding):
        y = mx_driving._C.npu_max_pool2d(x, kernel_size, stride, padding)
        return y


def npu_max_pool2d(x, kernel_size, stride, padding):
    DEVICE_NAME = torch_npu.npu.get_device_name(x.device.index)
    if "Ascend910" in DEVICE_NAME:
        return MaxPool2d.apply(x, kernel_size, stride, padding)
    elif "Ascend950" in DEVICE_NAME:
        return torch.nn.functional.max_pool2d(x, kernel_size, stride, padding)
    else:
        raise NotImplementedError("The npu_max_pool2d currently only supports Ascend910B, Ascend910C and Ascend950.")
