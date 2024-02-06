// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/framework/OpCommand.h"
#include "functions.h"
#include "common.h"


namespace {
at::Tensor &stride_add_out_npu_nocheck(
    at::Tensor &result,
    const at::Tensor &self,
    const at::Tensor &other,
    c10::Scalar offset1,
    c10::Scalar offset2,
    c10::Scalar c1_len)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("StrideAdd")
        .Input(self, "x1")
        .Input(other, "x2")
        .Output(result, "y")
        .Attr("x1_c1_offset", (int64_t)offset1.toInt())
        .Attr("x2_c1_offset", (int64_t)offset2.toInt())
        .Attr("c1_len", (int64_t)c1_len.toInt())
        .Run();
    return result;
}
} // namespace

at::Tensor npu_stride_add_compute(
    const at::Tensor &self,
    const at::Tensor &other,
    const c10::Scalar &offset1,
    const c10::Scalar &offset2,
    const c10::Scalar &c1_len)
{
    auto output_size = infersize_stride_add(self.sizes(), other.sizes());
    output_size[1] = c1_len.toInt() * 16;
    at::Tensor result = at::empty(output_size, self.options());
    stride_add_out_npu_nocheck(result, self, other, offset1, offset2, c1_len);
    return result;
}

at::Tensor npu_stride_add(py::args args)
{
    TORCH_CHECK(args.size() == 5U, "input arg size: ", args.size(), " is wrong, size should be 5");
    at::Tensor self = py::cast<at::Tensor>(args[0]);
    at::Tensor other = py::cast<at::Tensor>(args[1]);
    int offsetTmp1 = py::cast<int>(args[2]);
    int offsetTmp2 = py::cast<int>(args[3]);
    int lenTmp = py::cast<int>(args[4]);
    c10::Scalar offset1 = c10::Scalar(offsetTmp1);
    c10::Scalar offset2 = c10::Scalar(offsetTmp2);
    c10::Scalar c10Len = c10::Scalar(lenTmp);
    return npu_stride_add_compute(self, other, offset1, offset2, c10Len);
}
