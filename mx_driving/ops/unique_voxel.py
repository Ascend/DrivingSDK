import torch

import mx_driving._C


class UniqueVoxelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voxels):
        num_voxels_, uni_voxels, uni_indices, argsort_indices, uni_argsort_indices = mx_driving._C.unique_voxel(voxels)
        return num_voxels_, uni_voxels, uni_indices, argsort_indices, uni_argsort_indices
    
unique_voxel = UniqueVoxelFunction.apply