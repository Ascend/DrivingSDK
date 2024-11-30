"""
Copyright (c) OpenMMLab. All rights reserved.
Modification by: Huawei Developers
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification date: 2024-06-04 
Modification Description: 
Modification 1. Add support for Ascend NPU
"""
import torch
from torch.autograd import Function

import torch_npu
import mx_driving._C


class Nms3dFunction(Function):
    @staticmethod
    def forward(ctx, boxes, scores, iou_threshold: float):
        if boxes.shape[1] != 7:
            raise Exception('Input boxes shape should be (N, 7)')
        order = scores.sort(0, descending=True)[1]
        boxes = boxes[order].contiguous()

        keep, num_out = mx_driving._C.nms3d(boxes, iou_threshold)
        return order[keep[:num_out].long()].contiguous()


nms3d = Nms3dFunction.apply
npu_nms3d = Nms3dFunction.apply
