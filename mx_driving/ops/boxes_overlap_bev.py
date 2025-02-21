import warnings

import torch

import mx_driving._C


class BoxesOverlapBev(torch.autograd.Function):
    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx, boxes_a, boxes_b, inp_format, r_unit, clockwise, mode, aligned, margin):
        
        # input format
        format_dict = {
            "xyxyr": 0,     # [x1, y1, x2, y2, angle]
            "xywhr": 1,     # [x_center, y_center, dx, dy, angle]
            "xyzxyzr": 2,   # [x1, y1, z1, x2, y2, z2, angle]
            "xyzwhdr": 3    # [x_center, y_center, z_center, dx, dy, dz, angle]
        }
        format_flag = 0
        if inp_format in format_dict:
            format_flag = format_dict[inp_format]
        
        # rotate unit
        unit_dict = {
            "radian": 0,    # -pi ~ pi
            "degree": 1     # -180 ~ 180
        }
        unit_flag = 0
        if r_unit in unit_dict:
            unit_flag = unit_dict[r_unit]
        
        # calculation mode
        mode_dict = {
            "overlap": 0,   # area_overlap
            "iou": 1,       # area_overlap / (area_a + area_b - area_overlap)
            "iof": 2        # area_overlap / area_a
        }
        mode_flag = 0
        if mode in mode_dict:
            mode_flag = mode_dict[mode]
        
        # calculate result
        res = mx_driving._C.npu_boxes_overlap_bev(boxes_a, boxes_b, format_flag, unit_flag, clockwise,
                                                  mode_flag, aligned, margin)
        return res


def boxes_overlap_bev(boxes_a, boxes_b):
    r_unit = "radian"
    aligned = False
    mode = "overlap"
    margin = 1e-5
    if boxes_a.shape[-1] == 5:
        # BEVFusion version of boxes_overlap_bev
        inp_format = "xyxyr"
        clockwise = False
    else:
        # OpenPCDet and mmcv version of boxes_overlap_bev
        inp_format = "xyzwhdr"
        clockwise = True
    return BoxesOverlapBev.apply(boxes_a, boxes_b, inp_format, r_unit, clockwise, mode, aligned, margin)


def npu_boxes_overlap_bev(boxes_a, boxes_b):
    warnings.warn("`npu_boxes_overlap_bev` will be deprecated in future. Please use `boxes_overlap_bev` instead.", DeprecationWarning)
    r_unit = "radian"
    aligned = False
    mode = "overlap"
    margin = 1e-5
    if boxes_a.shape[-1] == 5:
        # BEVFusion version of boxes_overlap_bev
        inp_format = "xyxyr"
        clockwise = False
    else:
        # OpenPCDet and mmcv version of boxes_overlap_bev
        inp_format = "xyzwhdr"
        clockwise = True
    return BoxesOverlapBev.apply(boxes_a, boxes_b, inp_format, r_unit, clockwise, mode, aligned, margin)


def boxes_iou_bev(boxes_a, boxes_b):
    # OpenPCDet version of boxes_iou_bev
    inp_format = "xyzwhdr"
    r_unit = "radian"
    clockwise = True
    mode = "iou"
    aligned = False
    margin = 1e-5
    return BoxesOverlapBev.apply(boxes_a, boxes_b, inp_format, r_unit, clockwise, mode, aligned, margin)