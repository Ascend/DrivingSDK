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
#include "functions.h"


at::Tensor group_points(const at::Tensor& points, const at::Tensor& idx, int b, int c, int n, int npoints, int nsample)
{
    // b, c, n, and npoints do not need to be passed into gatherv2,
    // b, c, n, and npoints are calculated inside the operator
    // gatherv2 operator in ascend needs to set axis to 0, dims is 0
    c10::SmallVector<int64_t, N> axis = {0};
    int64_t dim = 0;

    auto index = at::arange(0, b);
    index = index.view({-1, 1, 1});
    index = at::mul(index, n);
    at::Tensor indices = at::add(index, idx);
    indices = indices.view({-1});

    at::Tensor trans_features = points.transpose(1, 2);
    at::Tensor features = trans_features.contiguous();
    features = features.view({b * n, c});

    at::Tensor out = at::empty({b, c, npoints, nsample}, points.options());

    out = out.view({b * npoints * nsample, c});

    EXEC_NPU_CMD(aclnnGatherV2, features, dim, indices, out);

    at::Tensor output = out.view({b, npoints, nsample, c}).transpose(1, 3).transpose(2, 3);

    return output;
}