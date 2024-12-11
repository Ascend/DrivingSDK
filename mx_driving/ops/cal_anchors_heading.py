import torch

import mx_driving._C


class CalAnchorsHeading(torch.autograd.Function):
    @staticmethod
    def forward(ctx, anchors, origin_pos=None):
        if origin_pos is None:
            batch_size = anchors.shape[0]
            origin_pos = torch.zeros((batch_size, 2), dtype=torch.float32, device=anchors.device)
        
        heading = mx_driving._C.cal_anchors_heading(anchors, origin_pos)
        return heading

cal_anchors_heading = CalAnchorsHeading.apply