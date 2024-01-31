from typing import Union, Tuple
import torch
from torch.autograd import Function
from torch.nn import Module
import ads_c


class AdsVoxelizationFunction(Function):
    @staticmethod
    def forward(
        ctx,
        points,
        voxel_size,
        coors_range,
        max_points: int = -1,
        max_voxels: int = -1,
        deterministic: bool = True):

        if max_points != -1:
            print("ERROR: npu is only support dynamic voxelizaiton, please check max_num_point param \n")
            print("return the result with dynamic voxelization") 

        float_espolin = 1e-9
        if (voxel_size[0] < float_espolin or 
            voxel_size[1] < float_espolin or
            voxel_size[2] < float_espolin):
            print("ERROR: voxel size should larger than zero")

        # compute voxel size
        grid_x = round((coors_range[3] - coors_range[0]) / voxel_size[0])
        grid_y = round((coors_range[4] - coors_range[1]) / voxel_size[1])
        grid_z = round((coors_range[5] - coors_range[2]) / voxel_size[2])
        
        # create coors
        coors = points.new_zeros(size=(3, points.size(0)), dtype=torch.int)
        result = ads_c.dynamic_voxelization(points, coors, grid_x, grid_y, grid_z, 
                                            voxel_size[0], voxel_size[1], voxel_size[2],
                                            coors_range[0], coors_range[1], coors_range[2])
        return result

voxelization = AdsVoxelizationFunction.apply


class Voxelization(torch.nn.Module):
    def __init__(self,
                voxel_size,
                point_cloud_range,
                max_num_points: int,
                max_voxels: int = 20000,
                deterministic: bool = True):
        super().__init__()

        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points

    def forward(self, input: torch.Tensor):
        return voxelization(input, self.voxel_size, self.point_cloud_range,
                            self.max_num_points)

    


        
