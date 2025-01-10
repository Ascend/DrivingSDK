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

at::Tensor diff_iou_rotated_sort_vertices(const at::Tensor& vertices, const at::Tensor& mask,
    const at::Tensor& num_valid)
{
    TORCH_CHECK_NPU(vertices);
    TORCH_CHECK_NPU(mask);
    TORCH_CHECK_NPU(num_valid);
    TORCH_CHECK(vertices.dim() == 4, "vertices must be a 4D Tensor, but got: ", vertices.dim());
    TORCH_CHECK(mask.dim() == 3, "mask must be a 3D Tensor, but got: ", mask.dim());
    TORCH_CHECK(num_valid.dim() == 2, "num_valid must be a 2D Tensor, but got: ", num_valid.dim());

    uint32_t B = vertices.size(0);
    uint32_t N = vertices.size(1);

    at::Tensor sortedIdx = at::empty({B, N, 9}, num_valid.options());
    at::Tensor mask_fp = mask.to(at::kFloat);

    EXEC_NPU_CMD(aclnnDiffIouRotatedSortVertices, vertices, mask_fp, num_valid, sortedIdx);
    
    return sortedIdx;
}