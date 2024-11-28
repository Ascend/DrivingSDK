// Copyright (c) 2023-2024 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Copyright (c) 2023-2024 Huawei Technologies Co., Ltd
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

void npu_roiaware_pool3d_forward(const at::Tensor& rois, const at::Tensor& pts, const at::Tensor& pts_feature,
    at::Tensor& argmax, at::Tensor& pts_idx_of_voxels, at::Tensor& pooled_features, int32_t mode)
{
    at::Tensor rois_cast = rois;
    at::Tensor pts_cast = pts;
    at::Tensor pts_feature_cast = pts_feature;
    at::Tensor pooled_features_cast = pooled_features;

    auto dtype = rois.dtype();
    if (dtype == at::kHalf) {
        rois_cast = rois_cast.to(at::kFloat);
        pts_cast = pts_cast.to(at::kFloat);
        pts_feature_cast = pts_feature_cast.to(at::kFloat);
        pooled_features_cast = pooled_features_cast.to(at::kFloat);
    }

    uint32_t max_pts_each_voxel = pts_idx_of_voxels.size(4);
    uint32_t outx = pts_idx_of_voxels.size(1);
    uint32_t outy = pts_idx_of_voxels.size(2);
    uint32_t outz = pts_idx_of_voxels.size(3);

    EXEC_NPU_CMD(aclnnRoiawarePool3d, rois_cast, pts_cast, pts_feature_cast, mode, max_pts_each_voxel, outx, outy, outz,
        argmax, pts_idx_of_voxels, pooled_features_cast);

    if (dtype == at::kHalf) {
        pooled_features_cast = pooled_features_cast.to(at::kHalf);
    }

    pooled_features.copy_(pooled_features_cast);
}
