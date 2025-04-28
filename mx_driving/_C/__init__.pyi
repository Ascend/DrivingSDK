from typing import List, Optional, Tuple

import torch

def _init_op_api_so_path(so_path: str) -> None: ...
def knn(
    xyz: torch.Tensor, center_xyz: torch.Tensor, k: int, is_from_knn: bool
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def npu_three_interpolate(
    b: int, c: int, m: int, n: int, points: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor: ...
def npu_three_interpolate_backward(
    b: int, c: int, n: int, m: int, grad_out: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor: ...
def scatter_max_with_argmax_v2(
    updates: torch.Tensor, indices: torch.Tensor, out: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def npu_scatter_max_backward(
    x: torch.Tensor, segment_ids: torch.Tensor, num_segments: torch.Tensor
) -> torch.Tensor: ...
def npu_scatter(self: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, dim: int) -> torch.Tensor: ...
def npu_scatter_mean_grad(
    grad_out: torch.Tensor, index: torch.Tensor, count: torch.Tensor, dim: int
) -> torch.Tensor: ...
def npu_scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim: Optional[int] = None,
    dim_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def npu_sort_pairs(
    keys_in: torch.Tensor, values_in: torch.Tensor, dim: int, descending: bool
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def npu_hypot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...
def npu_hypot_grad(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor, out_grad: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def assign_score_withk(
    points: torch.Tensor,
    centers: torch.Tensor,
    scores: torch.Tensor,
    knn_idx: torch.Tensor,
    output: torch.Tensor,
    B: int,
    N: int,
    npoint: int,
    M: int,
    K: int,
    out_dim: int,
    aggregate: int,
) -> None: ...
def assign_score_withk_grad(
    grad_out: torch.Tensor,
    points: torch.Tensor,
    centers: torch.Tensor,
    scores: torch.Tensor,
    knn_idx: torch.Tensor,
    grad_points: torch.Tensor,
    grad_centers: torch.Tensor,
    grad_scores: torch.Tensor,
    B: int,
    N: int,
    npoint: int,
    M: int,
    K: int,
    out_dim: int,
    aggregate: int,
) -> None: ...
def npu_max_pool2d(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor: ...
def multi_scale_deformable_attn(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor: ...
def multi_scale_deformable_attn_backward(
    value: torch.Tensor,
    shape: torch.Tensor,
    level_start_index: torch.Tensor,
    location_trans: torch.Tensor,
    attn_weight_trans: torch.Tensor,
    grad_output: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def npu_add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...
def npu_add_relu_grad(self: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor: ...
def fused_bias_leaky_relu(x: torch.Tensor, bias: torch.Tensor, negative_slop: float, scale: float) -> torch.Tensor: ...
def deformable_aggregation(
    mc_ms_feat: torch.Tensor,
    spatial_shape: torch.Tensor,
    scale_start_index: torch.Tensor,
    sampling_location: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor: ...
def deformable_aggregation_backward(
    mc_ms_feat: torch.Tensor,
    spatial_shape: torch.Tensor,
    scale_start_index: torch.Tensor,
    sampling_location: torch.Tensor,
    weights: torch.Tensor,
    grad_output: torch.Tensor,
    grad_mc_ms_feat: torch.Tensor,
    grad_sampling_location: torch.Tensor,
    grad_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def deformable_conv2d(
    input: torch.Tensor,
    offset: torch.Tensor,
    weight: torch.Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
    deformable_groups: int,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def modulated_deformable_conv2d(
    input: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias_opt: Optional[torch.Tensor],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
    deformable_groups: int,
    with_bias: int,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def deformable_conv2d_backward(
    input: torch.Tensor,
    weight: torch.Tensor,
    offset: torch.Tensor,
    offset_output: torch.Tensor,
    grad_y: torch.Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
    deformable_groups: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def modulated_deformable_conv2d_backward(
    input: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias_opt: Optional[torch.Tensor],
    offset_output: torch.Tensor,
    grad_y: torch.Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
    deformable_groups: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
def multi_to_sparse(
    out_features: torch.Tensor,
    unique_indices_offset: torch.Tensor,
    sorted_idx_to_former_indices: torch.Tensor,
    outidx_pair: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def multi_to_sparse_v2(
    features: torch.Tensor,
    weight: torch.Tensor,
    unique_indices_offset: torch.Tensor,
    sorted_idx_to_former_indices: torch.Tensor,
    outidx_pair: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def npu_sparse_conv3d(
    indices: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    out_channel: int,
    outSpatialShape: Tuple[int, int, int],
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def npu_sparse_conv3d_grad(
    indices_offset: torch.Tensor,
    former_sorted_indices: torch.Tensor,
    feature: torch.Tensor,
    weight: torch.Tensor,
    grad: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def nms3d_normal(boxes: torch.Tensor, nms_overlap_thresh: float) -> Tuple[torch.Tensor, torch.Tensor]: ...
def nms3d(boxes: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]: ...
def nms3d_on_sight(boxes: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]: ...
def npu_rotated_overlaps(self: torch.Tensor, query_boxes: torch.Tensor, trans: bool) -> torch.Tensor: ...
def npu_rotated_iou(
    boxes: torch.Tensor,
    query_boxes: torch.Tensor,
    trans: bool,
    mode: int,
    is_cross: bool,
    v_threshold: float,
    e_threshold: float,
) -> torch.Tensor: ...
def npu_boxes_overlap_bev(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    format_flag: int,
    unit_flag: int,
    clockwise: bool,
    mode_flag: int,
    aligned: bool,
    margin: float,
) -> torch.Tensor: ...
def roi_align_rotated_v2_forward_npu(
    input: torch.Tensor,
    rois_map: torch.Tensor,
    output: torch.Tensor,
    spatial_scale: float,
    sampling_ratio: int,
    pooled_height: int,
    pooled_width: int,
    aligned: bool,
    clockwise: bool,
) -> None: ...
def npu_roi_align_rotated_grad_v2(
    input: torch.Tensor,
    rois: torch.Tensor,
    grad_output: torch.Tensor,
    pooled_height: int,
    pooled_width: int,
    spatial_scale: float,
    sampling_ratio: int,
    aligned: bool,
    clockwise: bool,
) -> torch.Tensor: ...
def npu_box_iou_quadri(boxes_a: torch.Tensor, boxes_b: torch.Tensor, mode_flag: int, aligned: bool) -> torch.Tensor: ...
def npu_box_iou_rotated(
    boxes_a: torch.Tensor, boxes_b: torch.Tensor, mode_flag: int, aligned: bool
) -> torch.Tensor: ...
def border_align(
    input: torch.Tensor, rois: torch.Tensor, output: torch.Tensor, pooled_size: int
) -> None: ...
def border_align_backward(
    grad_out: torch.Tensor, boxes: torch.Tensor, argmax_idx: torch.Tensor, pool_size: int, height: int, width: int
) -> torch.Tensor: ...
def npu_roiaware_pool3d_forward(
    rois: torch.Tensor,
    pts: torch.Tensor,
    pts_feature: torch.Tensor,
    argmax: torch.Tensor,
    pts_idx_of_voxels: torch.Tensor,
    pooled_features: torch.Tensor,
    mode: int,
) -> None: ...
def roiaware_pool3d_grad(
    pts_idx_of_voxels: torch.Tensor, argmax: torch.Tensor, grad_out: torch.Tensor, npoints: int, pool_method: int
) -> torch.Tensor: ...
def npu_points_in_box(boxes: torch.Tensor, pts: torch.Tensor) -> torch.Tensor: ...
def npu_points_in_box_all(boxes: torch.Tensor, pts: torch.Tensor) -> torch.Tensor: ...
def npu_roipoint_pool3d_forward(
    num_sampled_points: int, points: torch.Tensor, point_features: torch.Tensor, boxes3d: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def group_points(
    points: torch.Tensor, idx: torch.Tensor, b: int, c: int, n: int, npoints: int, nsample: int
) -> torch.Tensor: ...
def group_points_backward(
    grad_out: torch.Tensor, idx: torch.Tensor, b: int, c: int, n: int, npoints: int, nsample: int
) -> torch.Tensor: ...
def vec_pool_backward(
    grad_new_features: torch.Tensor, point_cnt_of_grid: torch.Tensor, grouped_idxs: torch.Tensor, n: int, num_c_in: int
) -> torch.Tensor: ...
def point_to_voxel(
    points: torch.Tensor, voxel_sizes: List[float], coor_ranges: List[float], layout: str
) -> torch.Tensor: ...
def voxel_to_point(
    voxels: torch.Tensor, voxel_sizes: List[float], coor_ranges: List[float], layout: str
) -> torch.Tensor: ...
def unique_voxel(voxels: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
def hard_voxelize(
    points: torch.Tensor, voxel_sizes: List[float], coor_ranges: List[float], max_points: int, max_voxels: int
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]: ...
def npu_bev_pool(
    feat: torch.Tensor,
    geom_feat: torch.Tensor,
    interval_lengths: torch.Tensor,
    interval_starts: torch.Tensor,
    b: int,
    d: int,
    h: int,
    w: int,
) -> torch.Tensor: ...
def npu_bev_pool_backward(
    grad_out: torch.Tensor,
    geom_feat: torch.Tensor,
    interval_lengths: torch.Tensor,
    interval_starts: torch.Tensor,
    b: int,
    d: int,
    h: int,
    w: int,
) -> torch.Tensor: ...
def npu_bev_pool_v2(
    depth: torch.Tensor,
    feat: torch.Tensor,
    ranks_depth: torch.Tensor,
    ranks_feat: torch.Tensor,
    ranks_bev: torch.Tensor,
    interval_lengths: torch.Tensor,
    interval_starts: torch.Tensor,
    b: int,
    d: int,
    h: int,
    w: int,
) -> torch.Tensor: ...
def npu_bev_pool_v2_backward(
    grad_out: torch.Tensor,
    depth: torch.Tensor,
    feat: torch.Tensor,
    ranks_depth: torch.Tensor,
    ranks_feat: torch.Tensor,
    ranks_bev: torch.Tensor,
    interval_lengths: torch.Tensor,
    interval_starts: torch.Tensor,
    b: int,
    d: int,
    h: int,
    w: int,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def furthest_point_sampling_with_dist(
    points_dist: torch.Tensor, nearest_temp: torch.Tensor, num_points: int
) -> torch.Tensor: ...
def npu_dynamic_scatter(
    feats: torch.Tensor,
    coors: torch.Tensor,
    prefix_sum_point_per_voxel: torch.Tensor,
    argsort_coor: torch.Tensor,
    num_voxels: int,
    reduce_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def npu_dynamic_scatter_grad(
    grad_point_feats: torch.Tensor,
    grad_voxel_feats: torch.Tensor,
    prefix_sum_point_per_voxel: torch.Tensor,
    argsort_coor: torch.Tensor,
    compare_mask: torch.Tensor,
    reduce_type: str,
) -> None: ...
def npu_furthest_point_sampling(
    point_xyz: torch.Tensor, nearset_temp: torch.Tensor, num_points: int
) -> torch.Tensor: ...
def voxel_pooling_train(
    inputFeatures: torch.Tensor,
    geom: torch.Tensor,
    outputFeatures: torch.Tensor,
    posMemo: torch.Tensor,
    batchSize: int,
    numPoints: int,
    numChannels: int,
    numVoxelX: int,
    numVoxelY: int,
    numVoxelZ: int,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def voxel_pool_train_backward(
    grad_out: torch.Tensor, posMemo: torch.Tensor, batchSize: int, numPoints: int, numChannels: int, h: int, w: int
) -> torch.Tensor: ...
def dynamic_voxelization(
    points: torch.Tensor,
    coors: torch.Tensor,
    grid_x: int,
    grid_y: int,
    grid_z: int,
    voxel_x: float,
    voxel_y: float,
    voxel_z: float,
    coors_min_x: float,
    coors_min_y: float,
    coorsMinZ: float,
) -> torch.Tensor: ...
def npu_bev_pool_v3(
    depth: Optional[torch.Tensor],
    feat: torch.Tensor,
    ranks_depth: Optional[torch.Tensor],
    ranks_feat: Optional[torch.Tensor],
    ranks_bev: torch.Tensor,
    B: int,
    D: int,
    H: int,
    W: int,
) -> torch.Tensor: ...
def npu_bev_pool_v3_backward(
    grad_out: torch.Tensor,
    depth: Optional[torch.Tensor],
    feat: torch.Tensor,
    ranks_depth: Optional[torch.Tensor],
    ranks_feat: Optional[torch.Tensor],
    ranks_bev: torch.Tensor,
    B: int,
    D: int,
    H: int,
    W: int,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]: ...
def cal_anchors_heading(
    anchors: torch.Tensor,
    origin_pos: Optional[torch.Tensor],
) -> torch.Tensor: ...
def diff_iou_rotated_2d(
    box1: torch.Tensor,
    box2: torch.Tensor
) -> torch.Tensor: ...
def grid_sampler2d_v2(
    input: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> torch.Tensor: ...
def grid_sampler2d_v2_backward(
        grad_output: torch.Tensor,
        input: torch.Tensor,
        grid: torch.Tensor,
        interpolation_mode: int,
        padding_mode: int,
        align_corners: bool,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def npu_batch_matmul(
    projection_mat: torch.Tensor,
    pts_extend: torch.Tensor,
) -> torch.Tensor: ...
def boxes_iou_bev(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
) -> torch.Tensor: ...
def cartesian_to_frenet(
    pt: torch.Tensor,
    poly_line: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def min_area_polygons(
    pointsets: torch.Tensor
) -> torch.Tensor: ...
def radius(
    x: torch.Tensor, y: torch.Tensor, ptr_x: torch.Tensor, 
    ptr_y: torch.Tensor, r: int, max_num_neighbors: int
) -> torch.Tensor: ...
__all__ = [
    "knn",
    "npu_three_interpolate",
    "npu_three_interpolate_backward",
    "npu_batch_matmul",
    "scatter_max_with_argmax_v2",
    "npu_scatter_max_backward",
    "npu_scatter",
    "npu_scatter_mean_grad",
    "npu_scatter_mean",
    "npu_sort_pairs",
    "npu_hypot",
    "npu_hypot_grad",
    "assign_score_withk",
    "assign_score_withk_grad",
    "npu_max_pool2d",
    "multi_scale_deformable_attn",
    "multi_scale_deformable_attn_backward",
    "npu_add_relu",
    "npu_add_relu_grad",
    "fused_bias_leaky_relu",
    "deformable_aggregation",
    "deformable_aggregation_backward",
    "deformable_conv2d",
    "modulated_deformable_conv2d",
    "deformable_conv2d_backward",
    "modulated_deformable_conv2d_backward",
    "nms3d_normal",
    "nms3d",
    "nms3d_on_sight",
    "npu_rotated_overlaps",
    "npu_rotated_iou",
    "npu_boxes_overlap_bev",
    "npu_points_in_box",
    "npu_points_in_box_all",
    "npu_roipoint_pool3d_forward",
    "group_points",
    "group_points_backward",
    "vec_pool_backward",
    "point_to_voxel",
    "voxel_pooling_train",
    "voxel_pool_train_backward",
    "dynamic_voxelization",
    "furthest_point_sampling_with_dist",
    "npu_dynamic_scatter",
    "npu_dynamic_scatter_grad",
    "npu_furthest_point_sampling",
    "npu_bev_pool_v3",
    "npu_bev_pool_v3_backward",
    "cal_anchors_heading",
    "diff_iou_rotated_2d",
    "boxes_iou_bev",
    "cartesian_to_frenet",
    "min_area_polygons",
    "radius",
]
