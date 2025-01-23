// Copyright (c) 2024 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
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

#include "csrc/OpApiCommon.h"
#include "csrc/functions.h"

at::Tensor deformable_aggregation(const at::Tensor& mc_ms_feat, const at::Tensor& spatial_shape,
    const at::Tensor& scale_start_index, const at::Tensor& sampling_location, const at::Tensor& weights)
{
    TORCH_CHECK_NPU(mc_ms_feat);
    TORCH_CHECK_NPU(spatial_shape);
    TORCH_CHECK_NPU(scale_start_index);
    TORCH_CHECK_NPU(sampling_location);
    TORCH_CHECK_NPU(weights);

    auto feat_size = mc_ms_feat.sizes();
    auto weights_size = weights.sizes();
    auto batch_size = feat_size[0];
    auto num_feat = feat_size[1];
    auto num_embeds = feat_size[2];
    auto num_anchors = weights_size[1];
    auto num_pts = weights_size[2];
    auto num_cams = weights_size[3];
    auto num_scale = weights_size[4];
    auto num_groups = weights_size[5];

    TORCH_CHECK(mc_ms_feat.dim() == 3, "mc_ms_feat.dim() must be 3, but got: ", mc_ms_feat.dim());
    TORCH_CHECK(spatial_shape.dim() == 3, "spatial_shape.dim() must be 3, but got: ", spatial_shape.dim());
    TORCH_CHECK(scale_start_index.dim() == 2, "scale_start_index.dim() must be 2, but got: ", scale_start_index.dim());
    TORCH_CHECK(sampling_location.dim() == 5, "sampling_location.dim() must be 5, but got: ", sampling_location.dim());
    TORCH_CHECK(weights.dim() == 6, "weights.dim() must be 6, but got: ", weights.dim());
    TORCH_CHECK(num_groups % 8 == 0, "weights.sizes()[5] must be multiple of 8, but got: ", num_groups);

    at::Tensor out = at::empty({batch_size, num_anchors, num_embeds}, mc_ms_feat.options());

    EXEC_NPU_CMD(aclnnDeformableAggregation, mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights,
        batch_size, num_feat, num_embeds, num_anchors, num_pts, num_cams, num_scale, num_groups, out);
    return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> deformable_aggregation_backward(const at::Tensor& mc_ms_feat,
    const at::Tensor& spatial_shape, const at::Tensor& scale_start_index, const at::Tensor& sampling_location,
    const at::Tensor& weights, const at::Tensor& grad_output, const at::Tensor& grad_mc_ms_feat,
    const at::Tensor& grad_sampling_location, const at::Tensor& grad_weights)
{
    TORCH_CHECK_NPU(mc_ms_feat);
    TORCH_CHECK_NPU(spatial_shape);
    TORCH_CHECK_NPU(scale_start_index);
    TORCH_CHECK_NPU(sampling_location);
    TORCH_CHECK_NPU(weights);
    TORCH_CHECK_NPU(grad_output);

    TORCH_CHECK(mc_ms_feat.dim() == 3, "mc_ms_feat.dim() must be 3, but got: ", mc_ms_feat.dim());
    TORCH_CHECK(spatial_shape.dim() == 3, "spatial_shape.dim() must be 3, but got: ", spatial_shape.dim());
    TORCH_CHECK(scale_start_index.dim() == 2, "scale_start_index.dim() must be 2, but got: ", scale_start_index.dim());
    TORCH_CHECK(sampling_location.dim() == 5, "sampling_location.dim() must be 5, but got: ", sampling_location.dim());
    TORCH_CHECK(weights.dim() == 6, "weights.dim() must be 6, but got: ", weights.dim());

    EXEC_NPU_CMD(aclnnDeformableAggregationGrad, mc_ms_feat, spatial_shape, scale_start_index, sampling_location,
        weights, grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights);
    return std::make_tuple(grad_mc_ms_feat, grad_sampling_location, grad_weights);
}
