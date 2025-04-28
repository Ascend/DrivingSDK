__all__ = [
    "RoIPointPool3d",
    "SparseConv3d",
    "SubMConv3d",
    "SparseConvTensor",
    "SparseModule",
    "SparseSequential",
    "Voxelization",
    "assign_score_withk",
    "bev_pool",
    "bev_pool_v2",
    "bev_pool_v3",
    "border_align",
    "box_iou_quadri",
    "box_iou_rotated",
    "boxes_overlap_bev",
    "npu_boxes_overlap_bev",
    "boxes_iou_bev",
    "deform_conv2d",
    "dynamic_scatter",
    "furthest_point_sample_with_dist",
    "furthest_point_sample_with_dist",
    "npu_fused_bias_leaky_relu",
    "geometric_kernel_attention",
    "grid_sampler2d_v2",
    "group_points",
    "hypot",
    "knn",
    "modulated_deform_conv2d",
    "multi_scale_deformable_attn",
    "npu_multi_scale_deformable_attn_function",
    "npu_nms3d_normal",
    "npu_add_relu",
    "npu_deformable_aggregation",
    "npu_batch_matmul",
    "deformable_aggregation",
    "npu_dynamic_scatter",
    "npu_max_pool2d",
    "npu_nms3d",
    "MultiScaleDeformableAttnFunction",
    "npu_points_in_box",
    "npu_points_in_box_all",
    "points_in_box",
    "points_in_boxes_all",
    "pixel_group",
    "roi_align_rotated",
    "roiaware_pool3d",
    "npu_rotated_iou",
    "npu_rotated_overlaps",
    "scatter_max",
    "scatter_mean",
    "three_interpolate",
    "three_nn",
    "npu_voxel_pooling_train",
    "voxelization",
    "cal_anchors_heading",
    "npu_gaussian",
    "npu_draw_gaussian_to_heatmap",
    "npu_assign_target_of_single_head",
    "diff_iou_rotated_2d",
    "nms3d_on_sight",
    "cartesian_to_frenet",
    "min_area_polygons",
    "radius",
]

import os

import mx_driving._C

from .modules.roi_point_pool_3d import RoIPointPool3d
from .modules.sparse_conv import SparseConv3d, SubMConv3d
from .modules.sparse_modules import SparseConvTensor, SparseModule, SparseSequential
from .modules.voxelization import Voxelization
from .ops.assign_score_withk import assign_score_withk
from .ops.bev_pool import bev_pool
from .ops.bev_pool_v2 import bev_pool_v2
from .ops.bev_pool_v3 import bev_pool_v3
from .ops.border_align import border_align
from .ops.box_iou import box_iou_quadri, box_iou_rotated
from .ops.boxes_overlap_bev import boxes_overlap_bev, npu_boxes_overlap_bev, boxes_iou_bev
from .ops.deform_conv2d import DeformConv2dFunction, deform_conv2d
from .ops.furthest_point_sampling import furthest_point_sampling
from .ops.furthest_point_sampling_with_dist import furthest_point_sample_with_dist
from .ops.fused_bias_leaky_relu import npu_fused_bias_leaky_relu
from .ops.group_points import group_points
from .ops.geometric_kernel_attention import geometric_kernel_attention
from .ops.grid_sampler2d_v2 import grid_sampler2d_v2
from .ops.hypot import hypot
from .ops.knn import knn
from .ops.modulated_deform_conv2d import ModulatedDeformConv2dFunction, modulated_deform_conv2d
from .ops.multi_scale_deformable_attn import (
    MultiScaleDeformableAttnFunction,
    multi_scale_deformable_attn,
    npu_multi_scale_deformable_attn_function,
)
from .ops.nms3d_normal import nms3d_normal
from .ops.npu_add_relu import npu_add_relu
from .ops.npu_deformable_aggregation import npu_deformable_aggregation, deformable_aggregation
from .ops.npu_dynamic_scatter import npu_dynamic_scatter, dynamic_scatter
from .ops.npu_max_pool2d import npu_max_pool2d
from .ops.nms3d import nms3d
from .ops.npu_points_in_box import npu_points_in_box, points_in_box
from .ops.npu_points_in_box_all import npu_points_in_box_all, points_in_boxes_all
from .ops.pixel_group import pixel_group
from .ops.roi_align_rotated import roi_align_rotated
from .ops.roiaware_pool3d import roiaware_pool3d
from .ops.roipoint_pool3d import roipoint_pool3d
from .ops.rotated_iou import npu_rotated_iou
from .ops.rotated_overlaps import npu_rotated_overlaps
from .ops.scatter_max import scatter_max
from .ops.scatter_mean import scatter_mean
from .ops.scatter_add import scatter_add
from .ops.three_interpolate import three_interpolate
from .ops.three_nn import three_nn
from .ops.voxel_pooling_train import npu_voxel_pooling_train
from .ops.voxelization import voxelization
from .ops.unique_voxel import unique_voxel
from .ops.cal_anchors_heading import cal_anchors_heading
from .ops.npu_gaussian import npu_gaussian
from .ops.npu_draw_gaussian_to_heatmap import npu_draw_gaussian_to_heatmap
from .ops.npu_assign_target_of_single_head import npu_assign_target_of_single_head
from .ops.diff_iou_rotated import diff_iou_rotated_2d
from .ops.npu_batch_matmul import npu_batch_matmul
from .ops.nms3d_on_sight import nms3d_on_sight
from .ops.cartesian_to_frenet import cartesian_to_frenet
from .patcher import default_patcher_builder, patch_mmcv_version
from .ops.radius import radius
from .ops.min_area_polygons import min_area_polygons


def _set_env():
    mx_driving_root = os.path.dirname(os.path.abspath(__file__))
    mx_driving_opp_path = os.path.join(mx_driving_root, "packages", "vendors", "customize")
    ascend_custom_opp_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    ascend_custom_opp_path = (
        mx_driving_opp_path if not ascend_custom_opp_path else mx_driving_opp_path + ":" + ascend_custom_opp_path
    )
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ascend_custom_opp_path

    mx_driving_op_api_so_path = os.path.join(mx_driving_opp_path, "op_api", "lib", "libcust_opapi.so")
    mx_driving._C._init_op_api_so_path(mx_driving_op_api_so_path)


_set_env()
