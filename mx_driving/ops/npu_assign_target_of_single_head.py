"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
Modification by: Huawei Developers
Modification date: 2025-03-04
Modification Description:
Modification 1. Add support for Ascend NPU
"""

import torch
import torch_npu
from torch.autograd import Function

import mx_driving._C


class AssignTargetOfSingleHead(Function):
    @staticmethod
    # pylint: disable=huawei-too-many-arguments
    def forward(
        ctx,
        boxes,
        cur_class_id,
        num_classes,
        out_size_factor,
        gaussian_overlap,
        min_radius,
        voxel_size,
        pc_range,
        feature_map_size,
        norm_bbox=True,
        with_velocity=True,
        flip_angle=False,
        max_objs=500,
    ):
        output = mx_driving._C.npu_assign_target_of_single_head(
            boxes,
            cur_class_id,
            num_classes,
            out_size_factor,
            gaussian_overlap,
            min_radius,
            voxel_size,
            pc_range,
            feature_map_size,
            norm_bbox,
            with_velocity,
            flip_angle,
            max_objs,
        )
        return output


# pylint: disable=too-many-arguments,huawei-too-many-arguments
def npu_assign_target_of_single_head(
    boxes,
    cur_class_id,
    num_classes,
    out_size_factor,
    gaussian_overlap,
    min_radius,
    voxel_size,
    pc_range,
    feature_map_size,
    norm_bbox=True,
    with_velocity=True,
    flip_angle=False,
    max_objs=500,
):
    return AssignTargetOfSingleHead.apply(
        boxes,
        cur_class_id,
        num_classes,
        out_size_factor,
        gaussian_overlap,
        min_radius,
        voxel_size,
        pc_range,
        feature_map_size,
        norm_bbox,
        with_velocity,
        flip_angle,
        max_objs,
    )
