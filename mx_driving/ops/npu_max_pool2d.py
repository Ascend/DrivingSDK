"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-06-04 
Modification Description: 
Modification 1. Add support for Ascend NPU
"""
from torch.autograd import Function
import mx_driving._C


class MaxPool2d(Function):
    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx, x, kernel_size, stride, padding):
        y = mx_driving._C.npu_max_pool2d(x, kernel_size, stride, padding)
        return y

npu_max_pool2d = MaxPool2d.apply
