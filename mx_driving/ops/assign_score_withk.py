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
from torch.nn import Module

import mx_driving._C


class AssignScoreWithkFunction(Function):
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    @staticmethod
    def forward(ctx, scores, point_features, center_features, knn_idx, aggregate):
        agg = {"sum": 0, "avg": 1, "max": 2}
        B, N, M, out_dim = point_features.size()
        _, npoint, K, _ = scores.size()
        # pylint: disable=too-many-boolean-expressions
        if (B == 0 or N == 0 or M == 0 or K == 0 or npoint == 0 or out_dim == 0):
            raise Exception("Error! Input shape can not contain zero! \n")
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
            agg_idx)

        ctx.save_for_backward(output, point_features, center_features, scores, knn_idx)
        ctx.agg = agg_idx
        return output
        
    @staticmethod
    def backward(ctx, grad_out):
        _, point_features, center_features, scores, knn_idx = ctx.saved_tensors
        agg = ctx.agg
        B, N, M, out_dim = point_features.size()
        _, npoint, K, _ = scores.size()
        # pylint: disable=too-many-boolean-expressions
        if (B == 0 or N == 0 or M == 0 or K == 0 or npoint == 0 or out_dim == 0):
            raise Exception("Error! Input shape can not contain zero! \n")
        grad_point_features = point_features.new_zeros(point_features.shape)
        grad_center_features = center_features.new_zeros(center_features.shape)
        grad_scores = scores.new_zeros(scores.shape)

        mx_driving._C.assign_score_withk_grad(
            grad_out.contiguous(),
            point_features.contiguous(),
            center_features.contiguous(),
            scores.contiguous(),
            knn_idx.contiguous(),
            grad_point_features,
            grad_center_features,
            grad_scores,
            B,
            N,
            npoint,
            M,
            K,
            out_dim,
            agg)

        return grad_scores, grad_point_features, grad_center_features, None, None

assign_score_withk = AssignScoreWithkFunction.apply
