"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-10-06 
Modification Description: 
Modification 1. Add support for Ascend NPU
"""
import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class AssignScoreWithkFunction(Function):
    @staticmethod
    def forward(ctx, *args):
        scores, point_features, center_features, knn_idx, aggregate = args
        agg = {'sum': 0, 'avg': 1, 'max': 2}
        B, N, M, out_dim = point_features.size()
        _, npoint, K, _ = scores.size()

        points_select = point_features.new_empty((B, npoint, K, M, out_dim))
        centers_select = center_features.new_empty((B, npoint, M, out_dim))
        for b in range(B):
            for p in range(npoint):
                points_select[b, p] = torch.index_select(point_features[b], 0, knn_idx[b, p])
                centers_select[b, p] = center_features[b, knn_idx[b, p, 0]]
        points_select = points_select.transpose(0, 1).transpose(3, 4).contiguous()
        centers_select = centers_select.transpose(0, 1).transpose(2, 3).contiguous()
        scores = scores.unsqueeze(3).repeat(1, 1, 1, out_dim, 1).transpose(0, 1).contiguous()
    
        agg_idx = 0 if aggregate not in agg.keys() else agg[aggregate]
        output = ads_c.assign_score_withk(
            points_select,
            centers_select,
            scores,
            knn_idx,
            agg_idx,
            B,
            N,
            npoint,
            M,
            K,
            out_dim
        )
        return output.permute(1, 3, 0, 2).contiguous()
assign_score_withk = AssignScoreWithkFunction.apply