import warnings

from .ops.npu_points_in_box import npu_points_in_box
from .ops.npu_points_in_box_all import npu_points_in_box_all
from .ops.npu_points_in_box_all import points_in_boxes_all
from .ops.roipoint_pool3d import roipoint_pool3d
from .modules.roi_point_pool_3d import RoIPointPool3d

warnings.warn(
    "This package is deprecated and will be removed in future. Please use `mx_driving.api` instead.", DeprecationWarning
)