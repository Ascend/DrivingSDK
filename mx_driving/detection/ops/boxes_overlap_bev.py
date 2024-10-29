import warnings

import torch

import mx_driving._C


class BoxesOverlapBev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, boxes_a, boxes_b):
        area_overlap = mx_driving._C.npu_boxes_overlap_bev(boxes_a, boxes_b)
        return area_overlap


def boxes_overlap_bev(boxes_a, boxes_b):
    return BoxesOverlapBev.apply(boxes_a, boxes_b)


def npu_boxes_overlap_bev(boxes_a, boxes_b):
    warnings.warn("`npu_boxes_overlap_bev` will be deprecated in future. Please use `boxes_overlap_bev` instead.", DeprecationWarning)
    return BoxesOverlapBev.apply(boxes_a, boxes_b)
