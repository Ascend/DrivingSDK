"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""

import torch
from torch.autograd import Function

import mx_driving._C


class _Voxelization(Function):
    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(
        ctx, points, voxel_size, coors_range, max_points: int = -1, max_voxels: int = -1, deterministic: bool = True
    ):
        empty_tensor = (
            len(points) == 0,
            len(voxel_size) == 0,
            len(coors_range) == 0,
        )
        if any(empty_tensor):
            raise Exception("Error! Input Tensor can not be a empty Tensor.\n")
        if max_points != -1 and max_voxels != -1:
            return mx_driving._C.hard_voxelize(points, voxel_size, coors_range, max_points, max_voxels)

        float_espolin = 1e-9
        if voxel_size[0] < float_espolin or voxel_size[1] < float_espolin or voxel_size[2] < float_espolin:
            print("ERROR: voxel size should larger than zero")

        # compute voxel size
        grid_x = round((coors_range[3] - coors_range[0]) / voxel_size[0])
        grid_y = round((coors_range[4] - coors_range[1]) / voxel_size[1])
        grid_z = round((coors_range[5] - coors_range[2]) / voxel_size[2])

        # create coors
        coors = points.new_zeros(size=(3, points.size(0)), dtype=torch.int)
        result = mx_driving._C.dynamic_voxelization(
            points,
            coors,
            grid_x,
            grid_y,
            grid_z,
            voxel_size[0],
            voxel_size[1],
            voxel_size[2],
            coors_range[0],
            coors_range[1],
            coors_range[2],
        )
        return result


voxelization = _Voxelization.apply
