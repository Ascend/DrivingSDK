"""
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import mx_driving._C


class MinAreaPolygonsFunction(Function):
    @staticmethod
    def forward(ctx, pointsets):
        if pointsets.shape[1] != 18:
            raise 'Input pointsets shape should be (N, 18)'
        result = mx_driving._C.min_area_polygons(pointsets)
        return result
min_area_polygons = MinAreaPolygonsFunction.apply