import warnings

from .modules.voxelization import Voxelization
from .ops.bev_pool import bev_pool
from .ops.bev_pool_v2 import bev_pool_v2
from .ops.bev_pool_v3 import bev_pool_v3
from .ops.furthest_point_sampling import npu_furthest_point_sampling, furthest_point_sampling
from .ops.furthest_point_sampling_with_dist import furthest_point_sample_with_dist
from .ops.group_points import group_points, npu_group_points
from .ops.npu_dynamic_scatter import npu_dynamic_scatter
from .ops.voxel_pooling_train import npu_voxel_pooling_train
from .ops.voxelization import voxelization

warnings.warn(
    "This package is deprecated and will be removed in future. Please use `mx_driving.api` instead.", DeprecationWarning
)