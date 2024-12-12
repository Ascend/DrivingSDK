import torch

import mx_driving._C


class BoxIouQuadri(torch.autograd.Function):
    @staticmethod
    def forward(ctx, boxes_a, boxes_b, mode, aligned):
        mode_dict = {"iou": 0, "iof": 1}
        mode_flag = 0
        if mode in mode_dict:
            mode_flag = mode_dict[mode]

        boxes_a = boxes_a.contiguous()
        boxes_b = boxes_b.contiguous()

        ious = mx_driving._C.npu_box_iou_quadri(boxes_a, boxes_b, mode_flag, aligned)
        return ious


class BoxIouRotated(torch.autograd.Function):
    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx, boxes_a, boxes_b, mode, aligned, clockwise):
        mode_dict = {"iou": 0, "iof": 1}
        mode_flag = 0
        if mode in mode_dict:
            mode_flag = mode_dict[mode]

        if not clockwise:
            flip_mat = boxes_a.new_ones(boxes_a.shape[-1])
            flip_mat[-1] = -1
            boxes_a = boxes_a * flip_mat
            boxes_b = boxes_b * flip_mat
        boxes_a = boxes_a.contiguous()
        boxes_b = boxes_b.contiguous()

        ious = mx_driving._C.npu_box_iou_rotated(boxes_a, boxes_b, mode_flag, aligned)
        return ious


def box_iou_quadri(boxes_a, boxes_b, mode='iou', aligned=False):
    return BoxIouQuadri.apply(boxes_a, boxes_b, mode, aligned)


def box_iou_rotated(boxes_a, boxes_b, mode='iou', aligned=False, clockwise=True):
    return BoxIouRotated.apply(boxes_a, boxes_b, mode, aligned, clockwise)
