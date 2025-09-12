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

constexpr uint32_t NEW_XYZ_DIM = 3;
constexpr uint32_t XYZ_DIM = 3;
constexpr uint32_t ROT_DIM = 3;

constexpr uint32_t POINT_DIM = 3;
constexpr uint32_t ROT_SIZE = 9;

constexpr uint32_t B_INDEX = 0;
constexpr uint32_t N_INDEX = 1;
constexpr uint32_t M_INDEX = 1;
constexpr uint32_t INPUT_XYZ_DIM_INDEX = 2;
constexpr uint32_t INPUT_NEW_XYZ_DIM_INDEX = 2;
constexpr uint32_t INPUT_ROT_DIM_INDEX = 2;

at::Tensor cylinder_query(double radius, double hmin, double hmax, int nsample, const at::Tensor& new_xyz,
    const at::Tensor& xyz, const at::Tensor& rot)
{
    TORCH_CHECK_NPU(new_xyz);
    TORCH_CHECK_NPU(xyz);
    TORCH_CHECK_NPU(rot);
    TORCH_CHECK(new_xyz.dim() == NEW_XYZ_DIM, "new_xyz must be a 3D Tensor, but got: ", new_xyz.dim());
    TORCH_CHECK(xyz.dim() == XYZ_DIM, "xyz must be a 3D Tensor, but got: ", xyz.dim());
    TORCH_CHECK(rot.dim() == ROT_DIM, "rot must be a 3D Tensor, but got: ", rot.dim());

    TORCH_CHECK(rot.size(B_INDEX) == new_xyz.size(B_INDEX), "The batch sizes of rot and new_xyz must be equal.");
    TORCH_CHECK(rot.size(B_INDEX) == xyz.size(B_INDEX), "The batch sizes of rot and xyz must be equal.");

    TORCH_CHECK(new_xyz.size(INPUT_NEW_XYZ_DIM_INDEX) == POINT_DIM, "new_xyz Coordinates should be represented by 3 numbers, bug got: ", new_xyz.size(INPUT_NEW_XYZ_DIM_INDEX));
    TORCH_CHECK(xyz.size(INPUT_XYZ_DIM_INDEX) == POINT_DIM, "xyz Coordinates should be represented by 3 numbers, bug got: ", xyz.size(INPUT_XYZ_DIM_INDEX));
    TORCH_CHECK(rot.size(INPUT_ROT_DIM_INDEX) == ROT_SIZE, "The size of the last dimension in rot should be 9, bug got: ", xyz.size(INPUT_ROT_DIM_INDEX));

    TORCH_CHECK(rot.size(M_INDEX) == new_xyz.size(M_INDEX), "The number of rot and new_xyz must be equal.");

    TORCH_CHECK(hmin < hmax, "The value of hmin needs to be less than the value of hmax.");
    TORCH_CHECK(nsample <= xyz.size(N_INDEX), "The value of nsample should be greater than the number of points in the tensor xyz.");
    TORCH_CHECK(nsample > 0, "The value of nsample should be greater than 0.");

    uint32_t B = static_cast<uint32_t>(new_xyz.size(B_INDEX));
    uint32_t N = static_cast<uint32_t>(xyz.size(N_INDEX));
    uint32_t M = static_cast<uint32_t>(new_xyz.size(M_INDEX));

    at::Tensor origin_index = at::arange(0, xyz.size(N_INDEX), new_xyz.options().dtype(at::kFloat));
    at::Tensor out = at::empty({B, M, N}, new_xyz.options());
    EXEC_NPU_CMD(aclnnCylinderQuery, new_xyz, xyz, rot, origin_index, B, N, M, radius, hmin, hmax, nsample, out);
    return out;
}