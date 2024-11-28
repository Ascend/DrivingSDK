"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""
import torch
import torch_npu
import mx_driving._C

npu_rotated_iou = mx_driving._C.npu_rotated_iou
