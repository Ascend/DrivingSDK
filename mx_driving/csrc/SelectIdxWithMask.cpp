// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

at::Tensor select_idx_with_mask(const at::Tensor& poly_line, const at::Tensor& min_idx, const at::Tensor& pt, const at::Tensor& back_idx)
{
    TORCH_CHECK_NPU(poly_line);
    TORCH_CHECK_NPU(min_idx);
    TORCH_CHECK_NPU(pt);
    TORCH_CHECK_NPU(back_idx);
    
    TORCH_CHECK(poly_line.dim() == 3, "poly_line must be a 3D Tensor, but got: ", poly_line.dim());
    TORCH_CHECK(min_idx.dim() == 2, "min_idx must be a 2D Tensor, but got: ", min_idx.dim());
    TORCH_CHECK(pt.dim() == 3, "pt must be a 3D Tensor, but got: ", pt.dim());
    TORCH_CHECK(back_idx.dim() == 2, "back_idx must be a 2D Tensor, but got: ", back_idx.dim());

    TORCH_CHECK(poly_line.size(2) == 2, "The third dimension of poly_line must be 2, but got: ", poly_line.size(2));
    TORCH_CHECK(pt.size(1) == min_idx.size(1), "The second dimension of pt must match the second dimension of min_idx, but got: ", pt.size(1));
    TORCH_CHECK(back_idx.size(1) == min_idx.size(1), "The second dimension of back_idx must match the second dimension of min_idx, but got: ", back_idx.size(1));

    uint32_t batch_size = min_idx.size(0);
    uint32_t point_num = min_idx.size(1);

    at::Tensor  out_min_idx = at::empty({min_idx.size(0), min_idx.size(1)}, min_idx.options());
    EXEC_NPU_CMD(aclnnSelectIdxWithMask, poly_line, min_idx, pt, back_idx, out_min_idx);
    return out_min_idx;
}