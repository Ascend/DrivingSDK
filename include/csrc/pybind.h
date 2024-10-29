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
#ifndef CSRC_PYBIND_H_
#define CSRC_PYBIND_H_
#include <pybind11/pybind11.h>

void init_common(pybind11::module& m);
void init_fused(pybind11::module& m);
void init_point(pybind11::module& m);
void init_preprocess(pybind11::module& m);
void init_detection(pybind11::module& m);
void init_spconv(pybind11::module& m);
#endif // CSRC_PYBIND_H_
