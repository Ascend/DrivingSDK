"""
Copyright (c) OpenMMLab. All rights reserved.
Modification by: Huawei Developers
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
Modification date: 2025-01-15 
Modification Description: 
Modification 1. Add support for Ascend NPU
"""
import torch
from torch.autograd import Function

import torch_npu
import mx_driving._C


class Nms3dOnSightFunction(Function):
    @staticmethod
    def forward(ctx, boxes, scores, threshold: float):
        if boxes.shape[1] != 7:
            raise Exception('Input boxes shape should be (N, 7)')
        order = scores.sort(0, descending=True)[1]
        boxes = boxes[order].contiguous()

        keep, num_out = mx_driving._C.nms3d_on_sight(boxes, -threshold**2)
        return order[keep[:num_out].long()].contiguous()


nms3d_on_sight = Nms3dOnSightFunction.apply
npu_nms3d_on_sight = Nms3dOnSightFunction.apply
