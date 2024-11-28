"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-10-06 
Modification Description: 
Modification 1. Add support for Ascend NPU
"""

import torch
import torch_npu
from torch.autograd import Function

import mx_driving._C


class AssignScoreWithkFunction(Function):
    @staticmethod
    def forward(ctx, *args):
        scores, point_features, center_features, knn_idx, aggregate = args
        agg = {"sum": 0, "avg": 1, "max": 2}
        B, N, M, out_dim = point_features.size()
        _, npoint, K, _ = scores.size()
        agg_idx = 0 if aggregate not in agg.keys() else agg[aggregate]
        output = point_features.new_zeros((B, out_dim, npoint, K))
        mx_driving._C.assign_score_withk(
            point_features.contiguous(),
            center_features.contiguous(),
            scores.contiguous(),
            knn_idx.contiguous(),
            output,
            B,
            N,
            npoint,
            M,
            K,
            out_dim,
            agg_idx
        )
        return output
assign_score_withk = AssignScoreWithkFunction.apply
