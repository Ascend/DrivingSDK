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

#include "csrc/pybind.h"

#include <torch/extension.h>

#include "functions.h"

void init_common(pybind11::module& m)
{
    // knn
    m.def("knn", &knn);

    // npu_scatter_mean_grad
    m.def("npu_scatter_mean_grad", &npu_scatter_mean_grad);

    // three_interpolate
    m.def("npu_three_interpolate", &npu_three_interpolate);
    m.def("npu_three_interpolate_backward", &npu_three_interpolate_backward);

    // scatter_mean
    m.def("npu_scatter_mean", &npu_scatter_mean, "npu_scatter_mean NPU version");
    
    // scatter_max
    m.def("scatter_max_with_argmax_v2", &scatter_max_with_argmax_v2);
    m.def("npu_scatter_max_backward", &npu_scatter_max_backward);

    // npu_sort_pairs
    m.def("npu_sort_pairs", &npu_sort_pairs, "sort_pairs NPU version");
}
