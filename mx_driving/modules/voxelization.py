import torch
from torch.nn import Module
from torch.nn.modules.utils import _pair

from ..ops.voxelization import voxelization


class Voxelization(Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels=20000, deterministic=True):
        super().__init__()

        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.max_voxels = max_voxels if isinstance(max_voxels, tuple) else _pair(max_voxels)
        self.deterministic = deterministic

    def forward(self, points: torch.Tensor):
        max_voxels = self.max_voxels[0] if self.training else self.max_voxels[1]
        return voxelization(
            points, self.voxel_size, self.point_cloud_range, self.max_num_points, max_voxels, self.deterministic
        )
