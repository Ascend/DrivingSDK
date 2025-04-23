"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2025-02-26
Modification Description:
Modification 1. Add support for Ascend NPU
"""

from typing import Tuple, Union
import torch
from torch.autograd import Function
from torch.nn import Module
from torch import Tensor
import torch_npu
import mx_driving._C


class Radius(Function):
    @staticmethod
    # pylint: disable=huawei-too-many-arguments
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, ptr_x: torch.Tensor,
                ptr_y: torch.Tensor, r: float, max_num_neighbors: int) -> Tensor:
        output, actual_num_neighbors = mx_driving._C.radius(x, y, ptr_x, ptr_y, r, max_num_neighbors)
        return output[:, 0:actual_num_neighbors[0]]

radius = Radius.apply