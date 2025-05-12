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


class UniqueFunction(Function):
    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx, input_tensor):
        y = mx_driving._C.npu_unique(input_tensor)
        return y


npu_unique = UniqueFunction.apply