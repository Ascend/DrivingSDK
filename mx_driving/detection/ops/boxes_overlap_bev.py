import ads_c
import torch


class BoxesOverlapBev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, boxes_a, boxes_b):
        area_overlap = ads_c.npu_boxes_overlap_bev(boxes_a, boxes_b)
        return area_overlap

npu_boxes_overlap_bev = BoxesOverlapBev.apply
