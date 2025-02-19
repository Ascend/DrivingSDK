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

std::tuple<at::Tensor, at::Tensor, at::Tensor> calc_poly_start_end_sl(const at::Tensor& min_idx, const at::Tensor& poly_line, const at::Tensor& points, const at::Tensor& s_cum)
{
    TORCH_CHECK_NPU(min_idx);
    TORCH_CHECK_NPU(poly_line);
    TORCH_CHECK_NPU(points);
    TORCH_CHECK_NPU(s_cum);
    TORCH_CHECK(min_idx.dim() == 2, "min_idx.dim() must be 2, but got: ", min_idx.dim());
    TORCH_CHECK(poly_line.dim() == 3, "poly_line.dim() must be 3, but got: ", poly_line.dim());
    TORCH_CHECK(points.dim() == 3, "points.dim() must be 3, but got: ", points.dim());
    TORCH_CHECK(s_cum.dim() == 2, "s_cum.dim() must be 2, but got: ", s_cum.dim());
    at::Tensor poly_start = at::empty({points.size(0), points.size(1), points.size(2)}, points.options());
    at::Tensor poly_end = at::empty({points.size(0), points.size(1), points.size(2)}, points.options());
    at::Tensor sl = at::empty({points.size(0), points.size(1), points.size(2)}, points.options());
    EXEC_NPU_CMD(aclnnCalcPolyStartEndSl, min_idx, poly_line, points, s_cum, poly_start, poly_end, sl);
    return std::tie(poly_start, poly_end, sl);
}
