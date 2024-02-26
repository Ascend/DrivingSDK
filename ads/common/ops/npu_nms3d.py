import torch
from torch.autograd import Function

import torch_npu
import ads_c


class Nms3dFunction(Function):
    @staticmethod
    def forward(ctx, boxes, scores, iou_threshold: float):
        if boxes.shape[1] != 7:
            raise Exception('Input boxes shape should be (N, 7)')
        order = scores.sort(0, descending=True)[1]
        boxes = boxes[order].contiguous()

        keep, num_out = ads_c.nms3d(boxes, iou_threshold)
        return order[keep[:num_out].long()].contiguous()


npu_nms3d = Nms3dFunction.apply
