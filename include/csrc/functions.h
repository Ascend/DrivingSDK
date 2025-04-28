// Copyright (c) 2024, Huawei Technologies.All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef CSRC_FUNCTIONS_H_
#define CSRC_FUNCTIONS_H_

#include <ATen/ATen.h>

std::tuple<at::Tensor, at::Tensor> knn(
    const at::Tensor& xyz, const at::Tensor& center_xyz, int32_t k, bool is_from_knn);

at::Tensor npu_three_interpolate(
    int b, int c, int m, int n, const at::Tensor& points, const at::Tensor& idx, const at::Tensor& weight);

at::Tensor npu_three_interpolate_backward(
    int b, int c, int n, int m, const at::Tensor& grad_out, const at::Tensor& idx, const at::Tensor& weight);

std::tuple<at::Tensor, at::Tensor> scatter_max_with_argmax_v2(
    const at::Tensor& updates, const at::Tensor& indices, c10::optional<at::Tensor> out);

at::Tensor npu_scatter_max_backward(const at::Tensor& x, const at::Tensor& segment_ids, const at::Tensor& num_segments);

at::Tensor npu_scatter(const at::Tensor& self, const at::Tensor& indices, const at::Tensor& updates, int64_t dim);

at::Tensor npu_scatter_mean_grad(at::Tensor& grad_out, at::Tensor& index, at::Tensor& count, int32_t dim);

std::tuple<at::Tensor, at::Tensor> npu_scatter_mean(at::Tensor& src, at::Tensor& index, c10::optional<at::Tensor> out,
    c10::optional<int> dim, c10::optional<int> dim_size);
std::tuple<at::Tensor, at::Tensor> npu_sort_pairs(
    const at::Tensor& keys_in, const at::Tensor& values_in, int64_t dim, bool descending);

at::Tensor npu_hypot(const at::Tensor& x, const at::Tensor& y);

std::tuple<at::Tensor, at::Tensor> npu_hypot_grad(
    const at::Tensor& x, const at::Tensor& y, const at::Tensor& out, const at::Tensor& out_grad);

void assign_score_withk(const at::Tensor& points, const at::Tensor& centers, const at::Tensor& scores,
    const at::Tensor& knn_idx, at::Tensor& output, int32_t B, int32_t N, int32_t npoint, int32_t M, int32_t K,
    int32_t out_dim, int32_t aggregate);

void assign_score_withk_grad(const at::Tensor& grad_out, const at::Tensor& points, const at::Tensor& centers, const at::Tensor& scores,
    const at::Tensor& knn_idx, at::Tensor& grad_points, at::Tensor& grad_centers, at::Tensor& grad_scores,
    int32_t B, int32_t N, int32_t npoint, int32_t M, int32_t K, int32_t out_dim, int32_t aggregate);

at::Tensor npu_max_pool2d(const at::Tensor& x, int kernel_size, int stride, int padding);

at::Tensor multi_scale_deformable_attn(const at::Tensor& value, const at::Tensor& value_spatial_shapes,
    const at::Tensor& value_level_start_index, const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights);

std::tuple<at::Tensor, at::Tensor, at::Tensor> multi_scale_deformable_attn_backward(const at::Tensor& value,
    const at::Tensor& value_spatial_shapes, const at::Tensor& value_level_start_index,
    const at::Tensor& sampling_locations, const at::Tensor& attention_weights, const at::Tensor& grad_output);

std::tuple<at::Tensor, at::Tensor, at::Tensor> multi_scale_deformable_attn_grad_v2(const at::Tensor& value,
    const at::Tensor& shape, const at::Tensor& level_start_index, const at::Tensor& location_trans,
    const at::Tensor& attn_weight_trans, const at::Tensor& grad_output);

at::Tensor npu_add_relu(at::Tensor& x, const at::Tensor& y);

at::Tensor npu_add_relu_grad(at::Tensor& self, at::Tensor& grad_output);
std::tuple<at::Tensor, at::Tensor> npu_scatter_mean(at::Tensor& src, at::Tensor& index, c10::optional<at::Tensor> out,
    c10::optional<int> dim, c10::optional<int> dim_size);

at::Tensor fused_bias_leaky_relu(const at::Tensor& x, const at::Tensor& bias, double negative_slop, double scale);

at::Tensor deformable_aggregation(const at::Tensor& mc_ms_feat, const at::Tensor& spatial_shape,
    const at::Tensor& scale_start_index, const at::Tensor& sampling_location, const at::Tensor& weights);
std::tuple<at::Tensor, at::Tensor, at::Tensor> deformable_aggregation_backward(const at::Tensor& mc_ms_feat,
    const at::Tensor& spatial_shape, const at::Tensor& scale_start_index, const at::Tensor& sampling_location,
    const at::Tensor& weights, const at::Tensor& grad_output, const at::Tensor& grad_mc_ms_feat,
    const at::Tensor& grad_sampling_location, const at::Tensor& grad_weights);

std::tuple<at::Tensor, at::Tensor> deformable_conv2d(const at::Tensor& input, const at::Tensor& offset,
    const at::Tensor& weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, int64_t groups, int64_t deformable_groups);

std::tuple<at::Tensor, at::Tensor> modulated_deformable_conv2d(const at::Tensor& input, const at::Tensor& offset,
    const at::Tensor& mask, const at::Tensor& weight, const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, int64_t deformable_groups, int64_t with_bias);

std::tuple<at::Tensor, at::Tensor, at::Tensor> deformable_conv2d_backward(const at::Tensor& input,
    const at::Tensor& weight, const at::Tensor& offset, const at::Tensor& offset_output, const at::Tensor& grad_y,
    at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, int64_t deformable_groups);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> modulated_deformable_conv2d_backward(
    const at::Tensor& input, const at::Tensor& offset, const at::Tensor& mask, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt, const at::Tensor& offset_output, const at::Tensor& grad_y,
    at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, int64_t deformable_groups, int64_t with_bias);

at::Tensor group_points(
    const at::Tensor& points, const at::Tensor& idx, int64_t b, int64_t c, int64_t n, int64_t npoints, int64_t nsample);

at::Tensor group_points_backward(const at::Tensor& grad_out, const at::Tensor& idx, int64_t b, int64_t c, int64_t n,
    int64_t npoints, int64_t nsample);

at::Tensor vec_pool_backward(const at::Tensor& grad_new_features, const at::Tensor& point_cnt_of_grid,
    const at::Tensor& grouped_idxs, const int64_t n, const int64_t num_c_in);

at::Tensor point_to_voxel(const at::Tensor& points, const std::vector<float> voxel_sizes,
    const std::vector<float> coor_ranges, const char* layout);

at::Tensor voxel_to_point(const at::Tensor& voxels, const std::vector<float> voxel_sizes,
    const std::vector<float> coor_ranges, const char* layout);

std::tuple<int32_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor> unique_voxel(const at::Tensor& voxels);

std::tuple<int32_t, at::Tensor, at::Tensor, at::Tensor> hard_voxelize(const at::Tensor& points,
    const std::vector<float> voxel_sizes, const std::vector<float> coor_ranges, int64_t max_points, int64_t max_voxels);

at::Tensor npu_bev_pool(const at::Tensor& feat, const at::Tensor& geom_feat, const at::Tensor& interval_lengths,
    const at::Tensor& interval_starts, int64_t b, int64_t d, int64_t h, int64_t w);
at::Tensor npu_bev_pool_backward(const at::Tensor& grad_out, const at::Tensor& geom_feat,
    const at::Tensor& interval_lengths, const at::Tensor& interval_starts, int64_t b, int64_t d, int64_t h, int64_t w);

at::Tensor npu_bev_pool_v2(const at::Tensor& depth, const at::Tensor& feat, const at::Tensor& ranks_depth,
    const at::Tensor& ranks_feat, const at::Tensor& ranks_bev, const at::Tensor& interval_lengths,
    const at::Tensor& interval_starts, int64_t b, int64_t d, int64_t h, int64_t w);
std::tuple<at::Tensor, at::Tensor> npu_bev_pool_v2_backward(const at::Tensor& grad_out, const at::Tensor& depth,
    const at::Tensor& feat, const at::Tensor& ranks_depth, const at::Tensor& ranks_feat, const at::Tensor& ranks_bev,
    const at::Tensor& interval_lengths, const at::Tensor& interval_starts, int64_t b, int64_t d, int64_t h, int64_t w);

at::Tensor furthest_point_sampling_with_dist(
    const at::Tensor& points_dist, const at::Tensor& nearest_temp, int32_t num_points);

std::tuple<at::Tensor, at::Tensor> npu_dynamic_scatter(const at::Tensor& feats, const at::Tensor& coors,
    const at::Tensor& prefix_sum_point_per_voxel, const at::Tensor& argsort_coor, int32_t num_voxels,
    const char* reduce_type);

void npu_dynamic_scatter_grad(at::Tensor& grad_point_feats, const at::Tensor& grad_voxel_feats,
    const at::Tensor& prefix_sum_point_per_voxel, const at::Tensor& argsort_coor, const at::Tensor& compare_mask,
    const char* reduce_type);

at::Tensor npu_furthest_point_sampling(const at::Tensor& point_xyz, const at::Tensor& nearset_temp, int32_t num_points);

std::tuple<at::Tensor&, at::Tensor&> voxel_pooling_train(const at::Tensor& inputFeatures, const at::Tensor& geom,
    at::Tensor& outputFeatures, at::Tensor& posMemo, int batchSize, int numPoints, int numChannels, int numVoxelX,
    int numVoxelY, int numVoxelZ);

at::Tensor voxel_pool_train_backward(const at::Tensor& grad_out, const at::Tensor& posMemo, const int64_t batchSize,
    const int64_t numPoints, const int64_t numChannels, const int64_t h, const int64_t w);

at::Tensor dynamic_voxelization(const at::Tensor& points, at::Tensor& coors, int grid_x, int grid_y, int grid_z,
    double voxel_x, double voxel_y, double voxel_z, double coors_min_x, double coors_min_y, double coorsMinZ);

at::Tensor npu_bev_pool_v3(const c10::optional<at::Tensor>& depth, const at::Tensor& feat,
    const c10::optional<at::Tensor>& ranks_depth, const c10::optional<at::Tensor>& ranks_feat,
    const at::Tensor& ranks_bev, int64_t b, int64_t d, int64_t h, int64_t w);
std::tuple<c10::optional<at::Tensor>, at::Tensor> npu_bev_pool_v3_backward(const at::Tensor& grad_out,
    const c10::optional<at::Tensor>& depth, const at::Tensor& feat, const c10::optional<at::Tensor>& ranks_depth,
    const c10::optional<at::Tensor>& ranks_feat, const at::Tensor& ranks_bev);
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_subm_sparse_conv3d(const at::Tensor& feature,
    const at::Tensor& indices, const at::Tensor& weight, at::IntArrayRef kernel_size, int out_channel,
    at::IntArrayRef outSpatialShape, int batch_size, const at::Tensor& temp);

std::tuple<at::Tensor, at::Tensor> multi_to_sparse(const at::Tensor& out_features,
    const at::Tensor& unique_indices_offset, const at::Tensor& sorted_idx_to_former_indices,
    const at::Tensor& outidx_pair);

std::tuple<at::Tensor, at::Tensor> multi_to_sparse_v2(const at::Tensor& features, const at::Tensor& weight,
    const at::Tensor& unique_indices_offset, const at::Tensor& sorted_idx_to_former_indices,
    const at::Tensor& outidx_pair);

std::tuple<at::Tensor, at::Tensor> npu_sparse_conv3d(const at::Tensor& indices, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, int out_channel, at::IntArrayRef outSpatialShape, int batch_size);

std::tuple<at::Tensor, at::Tensor> npu_sparse_conv3d_grad(const at::Tensor& indices_offset,
    const at::Tensor& former_sorted_indices, const at::Tensor& feature, const at::Tensor& weight,
    const at::Tensor& grad);

std::tuple<at::Tensor, at::Tensor> npu_prepare_subm_conv3d(
    const at::Tensor& flattenIndices, at::IntArrayRef outSpatialShape, int batch_size);

std::tuple<at::Tensor, at::Tensor> nms3d_normal(const at::Tensor& boxes, double nms_overlap_thresh);

std::tuple<at::Tensor, at::Tensor> nms3d(const at::Tensor& boxes, double threshold);

std::tuple<at::Tensor, at::Tensor> nms3d_on_sight(const at::Tensor& boxes, double threshold);

at::Tensor npu_rotated_overlaps(const at::Tensor& self, const at::Tensor& query_boxes, bool trans);

at::Tensor npu_rotated_iou(const at::Tensor& boxes, const at::Tensor& query_boxes, bool trans, int64_t mode,
    bool is_cross, double v_threshold, double e_threshold);

at::Tensor npu_boxes_overlap_bev(const at::Tensor& boxes_a, const at::Tensor& boxes_b,
    int32_t format_flag, int32_t unit_flag, bool clockwise, int32_t mode_flag, bool aligned, double margin);

void roi_align_rotated_v2_forward_npu(const at::Tensor& input, const at::Tensor& rois_map, at::Tensor& output,
    double spatial_scale, int32_t sampling_ratio, int32_t pooled_height, int32_t pooled_width, bool aligned,
    bool clockwise);
at::Tensor npu_roi_align_rotated_grad_v2(const at::Tensor& input, const at::Tensor& rois, const at::Tensor& grad_output,
    int32_t pooled_height, int32_t pooled_width, double spatial_scale, int32_t sampling_ratio, bool aligned,
    bool clockwise);

at::Tensor npu_box_iou_quadri(
    const at::Tensor& boxes_a, const at::Tensor& boxes_b, const int64_t mode_flag, const bool aligned);

at::Tensor npu_box_iou_rotated(
    const at::Tensor& boxes_a, const at::Tensor& boxes_b, const int64_t mode_flag, const bool aligned);

void border_align(const at::Tensor& input, const at::Tensor& rois, at::Tensor& output, int32_t pooled_size);

at::Tensor border_align_backward(const at::Tensor& grad_out, const at::Tensor& boxes, const at::Tensor& argmax_idx,
    int32_t pool_size, int32_t height, int32_t width);

void npu_roiaware_pool3d_forward(const at::Tensor& rois, const at::Tensor& pts, const at::Tensor& pts_feature,
    at::Tensor& argmax, at::Tensor& pts_idx_of_voxels, at::Tensor& pooled_features, int32_t mode);
at::Tensor roiaware_pool3d_grad(const at::Tensor& pts_idx_of_voxels, const at::Tensor& argmax,
    const at::Tensor& grad_out, int32_t npoints, int64_t pool_method);

std::vector<std::vector<float>> pixel_group(const at::Tensor& score, const at::Tensor& mask,
    const at::Tensor& embedding, const at::Tensor& kernel_label, const at::Tensor& kernel_contour,
    int kernel_region_num, double distance_threshold);

at::Tensor npu_points_in_box(const at::Tensor& boxes, const at::Tensor& pts);

at::Tensor npu_points_in_box_all(const at::Tensor& boxes, const at::Tensor& pts);

std::tuple<at::Tensor, at::Tensor> npu_roipoint_pool3d_forward(const int32_t num_sampled_points,
    const at::Tensor& points, const at::Tensor& point_features, const at::Tensor& boxes3d);

void geometric_kernel_attention_forward(const at::Tensor& value_map, const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index, const at::Tensor& sampling_locations,
    const at::Tensor& attention_weights, at::Tensor& output);

std::tuple<at::Tensor, at::Tensor> geometric_kernel_attention_backward(const at::Tensor& value,
    const at::Tensor& spatial_shapes, const at::Tensor& level_start_index, const at::Tensor& sampling_locations,
    const at::Tensor& attn_weights, const at::Tensor& grad_output);

at::Tensor cal_anchors_heading(const at::Tensor& anchors, const at::Tensor& origin_pos);
at::Tensor npu_subm_sparse_conv3d_grad(const at::Tensor& ouidx_offset, const at::Tensor& valid_indices,
                                       const at::Tensor& weight, const at::Tensor& grad, int indices_number,
                                       at::IntArrayRef kernel_size);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_gaussian(const at::Tensor& boxes,
    int32_t out_size_factor, float overlap, int32_t min_radius, float size_x, float size_y,
    float range_x, float range_y, int32_t feature_map_size_x, int32_t feature_map_size_y,
    bool norm_bbox, bool with_velocity, bool flip_angle, int32_t max_objs);

at::Tensor npu_draw_gaussian_to_heatmap(const at::Tensor& mask, const at::Tensor& cur_class_id, const at::Tensor& center_int, const at::Tensor& radius,
    int64_t feature_map_size_x, int64_t feature_map_size_y, int64_t num_classes);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_assign_target_of_single_head(const at::Tensor& boxes, const at::Tensor& cur_class_id,
    int32_t num_classes, int32_t out_size_factor, float overlap, int32_t min_radius,
    const std::vector<float> voxel_size, const std::vector<float> pc_range, at::IntArrayRef feature_map_size,
    bool norm_bbox, bool with_velocity, bool flip_angle, int32_t max_objs);

at::Tensor diff_iou_rotated_sort_vertices(const at::Tensor& vertices, const at::Tensor& mask,
    const at::Tensor& num_valid);

at::Tensor grid_sampler2d_v2(const at::Tensor& input, const at::Tensor& grid, int64_t interpolation_mode,
    int64_t padding_mode, bool align_corners);

std::tuple<at::Tensor, at::Tensor> grid_sampler2d_v2_backward(const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners);

at::Tensor npu_scatter_add(at::Tensor& src, at::Tensor& indices, c10::optional<at::Tensor> out,
    c10::optional<int> dim, c10::optional<int> dim_size);

at::Tensor npu_scatter_add_grad(at::Tensor& grad_out, at::Tensor& index, int32_t dim);

at::Tensor npu_batch_matmul(const at::Tensor& projection_mat, const at::Tensor& pts_extend);

at::Tensor select_idx_with_mask(const at::Tensor& poly_line, const at::Tensor& min_idx, const at::Tensor& pt, const at::Tensor& back_idx);

std::tuple<at::Tensor, at::Tensor> cartesian_to_frenet1(const at::Tensor& dist_vec);

std::tuple<at::Tensor, at::Tensor, at::Tensor> calc_poly_start_end_sl(const at::Tensor& min_idx, const at::Tensor& poly_line, const at::Tensor& points, const at::Tensor& s_cum);

at::Tensor npu_subm_sparse_conv3d_with_key(const at::Tensor& ouidx_offset, const at::Tensor& valid_indices,
                                           const at::Tensor& weight, const at::Tensor& feature, int indices_number,
                                           at::IntArrayRef kernel_size);

at::Tensor min_area_polygons(const at::Tensor& pointsets);

std::tuple<at::Tensor, at::Tensor> npu_subm_sparse_conv3d_v2(const at::Tensor& feature,
    const at::Tensor& indices, const at::Tensor& map1, const at::Tensor& map2, at::IntArrayRef kernel_size, int in_channels,
    at::IntArrayRef out_spatial_shape, int batch_size);

std::tuple<at::Tensor, at::Tensor> radius(at::Tensor& x, at::Tensor& y, at::Tensor& ptr_x, at::Tensor& ptr_y, double r, int max_num_neighbors);

#endif // CSRC_FUNCTIONS_H_
