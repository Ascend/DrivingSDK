"""
Copyright (c) Megvii Inc. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-06-04 
Modification Description: 
Modification 1. Add support for Ascend NPU
"""

import torch
from torch.autograd import Function

import mx_driving._C


class AdsVoxelPoolingFunction(Function):
    @staticmethod
    def forward(ctx, geom_xyz, input_features, voxel_num):
        grad_input_features = torch.zeros_like(input_features)
        geom_xyz = geom_xyz.reshape(geom_xyz.shape[0], -1, geom_xyz.shape[-1])
        input_features = input_features.reshape(geom_xyz.shape[0], -1, input_features.shape[-1])

        batch_size = input_features.shape[0]
        num_points = input_features.shape[1]
        if (num_points == 0):
            raise Exception("Error! Number of points can not be zero.\n")
        
        num_channels = input_features.shape[2]
        output_features = input_features.new_zeros(batch_size, voxel_num[1], voxel_num[0], num_channels)
        pos_memo = geom_xyz.new_ones(batch_size, num_points, 3) * -1
        pos, result = mx_driving._C.voxel_pooling_train(
            input_features,
            geom_xyz,
            output_features,
            pos_memo,
            batch_size,
            num_points,
            num_channels,
            voxel_num[0],
            voxel_num[1],
            voxel_num[2],
        )
        ctx.save_for_backward(grad_input_features, pos)
        return result.permute(0, 3, 1, 2)

    @staticmethod
    def backward(ctx, grad_output_features):
        (grad_input_features, pos_memo) = ctx.saved_tensors
        grad_input_features_shape = grad_input_features.shape

        batch_size = pos_memo.shape[0]
        num_points = pos_memo.shape[1]
        num_channels = grad_output_features.shape[1]
        H = grad_output_features.shape[2]
        W = grad_output_features.shape[3]

        result = mx_driving._C.voxel_pool_train_backward(
            grad_output_features, pos_memo, batch_size, num_points, num_channels, H, W
        )
        grad_input_features = result.reshape(grad_input_features_shape)
        return None, grad_input_features, None


npu_voxel_pooling_train = AdsVoxelPoolingFunction.apply
