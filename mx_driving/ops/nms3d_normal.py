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


class AdsNms3dNormalFunction(Function):
    @staticmethod
    def forward(ctx, boxes, scores, iou_threshold: float):
        if boxes.shape[1] != 7:
            raise "Input boxes shape should be (N, 7)"
        order = scores.sort(0, descending=True)[1]
        boxes = boxes[order].contiguous()

        keep, num_out = mx_driving._C.nms3d_normal(boxes, iou_threshold)
        return order[keep[:num_out].long()].contiguous()


nms3d_normal = AdsNms3dNormalFunction.apply
npu_nms3d_normal = AdsNms3dNormalFunction.apply
