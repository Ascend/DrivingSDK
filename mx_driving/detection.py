import warnings

from .ops.boxes_overlap_bev import boxes_overlap_bev, npu_boxes_overlap_bev
from .ops.nms3d_normal import nms3d_normal, npu_nms3d_normal
from .ops.nms3d import nms3d, npu_nms3d
from .ops.rotated_iou import npu_rotated_iou
from .ops.rotated_overlaps import npu_rotated_overlaps
from .ops.roi_align_rotated import roi_align_rotated
from .ops.box_iou import box_iou_quadri
from .ops.border_align import border_align
from .ops.roiaware_pool3d import roiaware_pool3d
from .ops.pixel_group import pixel_group

warnings.warn(
    "This package is deprecated and will be removed in future. Please use `mx_driving.api` instead.", DeprecationWarning
)