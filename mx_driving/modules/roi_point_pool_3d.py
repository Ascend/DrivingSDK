from torch.nn import Module

from ..ops.roipoint_pool3d import roipoint_pool3d


class RoIPointPool3d(Module):
    def __init__(self, num_sampled_points: int = 512):
        super().__init__()
        self.num_sampled_points = num_sampled_points

    def forward(self, points, point_features, boxes3d):
        return roipoint_pool3d(self.num_sampled_points, points, point_features, boxes3d)
