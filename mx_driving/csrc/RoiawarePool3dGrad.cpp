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


at::Tensor roiaware_pool3d_grad(const at::Tensor& pts_idx_of_voxels, const at::Tensor& argmax,
    const at::Tensor& grad_out, int32_t npoints, int64_t pool_method)
{
    TORCH_CHECK_NPU(pts_idx_of_voxels);
    TORCH_CHECK_NPU(argmax);
    TORCH_CHECK_NPU(grad_out);

    TORCH_CHECK(
        pts_idx_of_voxels.dim() == 5, "pts_idx_of_voxels must to be a 5D Tensor, but got: ", pts_idx_of_voxels.dim());
    TORCH_CHECK(argmax.dim() == 5, "argmax as to be a 5D Tensor, but got: ", argmax.dim());
    TORCH_CHECK(grad_out.dim() == 5, "grad_out has to be a 5D Tensor, but got: ", grad_out.dim());

    int32_t boxes_num = grad_out.size(0);
    int32_t out_x = grad_out.size(1);
    int32_t out_y = grad_out.size(2);
    int32_t out_z = grad_out.size(3);
    int32_t channels = grad_out.size(4);
    int32_t max_pts_per_voxel = pts_idx_of_voxels.size(4);

    TORCH_CHECK((boxes_num != 0 && out_x != 0 && out_y != 0 && out_z != 0 && channels != 0 && npoints != 0),
        "Error, some dim equals zero!\n");
    TORCH_CHECK((channels <= 2048), "channels must less equal than 2048, but got: ", channels);

    auto dtype = grad_out.dtype();
    at::Tensor grad_out_cast = grad_out;

    at::Tensor grad_in = at::zeros({npoints, channels}, grad_out.options());
    if (dtype == at::kHalf) {
        grad_out_cast = grad_out.to(at::kFloat);
        grad_in = grad_in.to(at::kFloat);
    }

    if (pool_method == 0) {
        // maxpool3d
        EXEC_NPU_CMD(aclnnRoiawareMaxpool3dGrad, argmax, grad_out_cast, boxes_num, out_x, out_y, out_z, channels,
            npoints, grad_in);
    } else if (pool_method == 1) {
        // avgpool3d
        TORCH_CHECK(npoints >= max_pts_per_voxel, "npoints must greator than max_pts_per_voxel!");
        TORCH_CHECK(max_pts_per_voxel != 0, "Error, some dim equals zero!");
        TORCH_CHECK(
            (max_pts_per_voxel <= 2048), "max_pts_per_voxel must less equal than 2048, but got: ", max_pts_per_voxel);

        EXEC_NPU_CMD(aclnnRoiawareAvgpool3dGrad, pts_idx_of_voxels, grad_out_cast, boxes_num, out_x, out_y, out_z,
            channels, npoints, max_pts_per_voxel, grad_in);
    }

    if (dtype == at::kHalf) {
        grad_in = grad_in.to(at::kHalf);
    }
    return grad_in;
}
