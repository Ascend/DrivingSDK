"""
Copyright (c) Megvii Inc. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-06-04 
Modification Description: 
Modification 1. Add support fro Ascend NPU
"""
import torch
from torch.autograd import Function
from torch.nn import Module
import ads_c


class AdsVoxelPoolingFunction(Function):
    @staticmethod
    def forward(ctx, geom_xyz, input_features, voxel_num):
        grad_input_features = torch.zeros_like(input_features)
        geom_xyz = geom_xyz.reshape(geom_xyz.shape[0], -1, geom_xyz.shape[-1])
        input_features = input_features.reshape(geom_xyz.shape[0], -1, input_features.shape[-1])
        
        batch_size = input_features.shape[0]
        num_points = input_features.shape[1]
        num_channels = input_features.shape[2]
        output_features = input_features.new_zeros(batch_size, voxel_num[1], 
                                                   voxel_num[0], num_channels)
        pos_memo = geom_xyz.new_ones(batch_size, num_points, 3) * -1
        pos, result = ads_c.voxel_pooling_train(
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
    
npu_voxel_pooling_train = AdsVoxelPoolingFunction.apply