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
#include "common.h"

at::Tensor npu_confusion_transpose(
    const at::Tensor& self,
    at::IntArrayRef perm,
    at::IntArrayRef shape,
    bool transpose_first)
{
    c10::SmallVector<int64_t, SIZE> output_size;
    if (transpose_first) {
        output_size = array_to_small_vector(shape);
    } else {
        auto shape_size = shape.size();
        for (uint i = 0; i < perm.size(); i++) {
            TORCH_CHECK(shape_size > perm[i], "npu_confusion_transpose input invalid, "
                                            "shape has size ",
                        shape_size, " but perm[i] is, ", perm[i]);
            output_size.emplace_back(shape[perm[i]]);
        }
    }

    at::Tensor result = at::empty(output_size, self.options());
    at_npu::native::OpCommand cmd;
    cmd.Name("ConfusionTransposeD")
        .Input(self)
        .Output(result)
        .Attr("perm", perm)
        .Attr("shape", shape)
        .Attr("transpose_first", transpose_first)
        .Run();

    return result;
}

void check_confusion_transpose_perm(at::IntArrayRef perm, at::IntArrayRef shape)
{
    auto input_dim = shape.size();
    TORCH_CHECK(perm.size() == input_dim, "The length of perm should be the same as shape.");
    std::vector<bool> seen(input_dim);
    for (const auto i : c10::irange(input_dim)) {
        auto dim = at::maybe_wrap_dim(perm[i], input_dim);
        TORCH_CHECK(!seen[dim], "Repeated dim in perm");
        seen[dim] = true;
    }
}

at::Tensor npu_confusion_transpose_backward(
    const at::Tensor& grad,
    at::IntArrayRef perm,
    at::IntArrayRef shape,
    bool transpose_first)
{
    c10::SmallVector<int64_t, SIZE> svec_shape;
    if (transpose_first) {
        svec_shape = array_to_small_vector(shape);
    } else {
        check_confusion_transpose_perm(perm, shape);
        for (int i = 0; i < perm.size(); i++) {
            svec_shape.emplace_back(shape[perm[i]]);
        }
    }
    std::vector<int64_t> vec_perm;
    int64_t perm_len = perm.size();
    int64_t temp_perm[perm_len] = {0};
    for (int64_t i = 0; i < perm_len; i++) {
        temp_perm[perm[i]] = i;
    }
    vec_perm = std::vector<int64_t>(temp_perm, temp_perm+perm_len);
    perm = at::IntArrayRef(vec_perm);
    at::Tensor result = at::empty(shape, grad.options());

    at_npu::native::OpCommand cmd;
    cmd.Name("ConfusionTransposeD")
        .Input(grad)
        .Output(result)
        .Attr("perm", perm)
        .Attr("shape", svec_shape)
        .Attr("transpose_first", transpose_first)
        .Run();
    return result;
}
