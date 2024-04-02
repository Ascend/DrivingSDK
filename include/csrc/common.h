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

#ifndef CSRC_COMMON_H_
#define CSRC_COMMON_H_
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <string>
#include <tuple>
#include <unordered_map>

#include "third_party/acl/inc/acl/acl_base.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"

const int N = 32;
const int SIZE = 8;

using tuple_vector = std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>;
using CalcuOpUtil = at_npu::native::CalcuOpUtil;

#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_)  \
    _(at::ScalarType::Byte, ACL_UINT8)               \
    _(at::ScalarType::Char, ACL_INT8)                \
    _(at::ScalarType::Short, ACL_INT16)              \
    _(at::ScalarType::Int, ACL_INT32)                \
    _(at::ScalarType::Long, ACL_INT64)               \
    _(at::ScalarType::Half, ACL_FLOAT16)             \
    _(at::ScalarType::Float, ACL_FLOAT)              \
    _(at::ScalarType::Double, ACL_DOUBLE)            \
    _(at::ScalarType::ComplexHalf, ACL_DT_UNDEFINED) \
    _(at::ScalarType::ComplexFloat, ACL_COMPLEX64)   \
    _(at::ScalarType::ComplexDouble, ACL_COMPLEX128) \
    _(at::ScalarType::Bool, ACL_BOOL)                \
    _(at::ScalarType::QInt8, ACL_DT_UNDEFINED)       \
    _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED)      \
    _(at::ScalarType::QInt32, ACL_DT_UNDEFINED)      \
    _(at::ScalarType::BFloat16, ACL_BF16)            \
    _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED)    \
    _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)    \
    _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)   \
    _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)

static std::unordered_map<std::string, at::ScalarType> dTypeTransMap {{"torch.float16", at::ScalarType::Half},
    {"torch.half", at::ScalarType::Half}, {"torch.float32", at::ScalarType::Float},
    {"torch.float", at::ScalarType::Float}, {"torch.float64", at::ScalarType::Double},
    {"torch.float", at::ScalarType::Double}, {"torch.int8", at::ScalarType::Char}, {"torch.char", at::ScalarType::Char},
    {"torch.int16", at::ScalarType::Short}, {"torch.short", at::ScalarType::Short},
    {"torch.int32", at::ScalarType::Int}, {"torch.int32", at::ScalarType::Int}, {"torch.int64", at::ScalarType::Long},
    {"torch.long", at::ScalarType::Long}};

inline static bool check_inplace_tensor(const std::initializer_list<at::Tensor>& src_list, const at::Tensor& dst)
{
    bool is_inplace_tensor = false;
    // check whether dst is contained in src_list
    for (const auto& src : src_list) {
        if (dst.is_same(src)) {
            is_inplace_tensor = true;
            break;
        }
    }
    return is_inplace_tensor;
}

inline static void check_tensor_size(
    const std::initializer_list<at::Tensor>& src_list, at::Tensor& dst, c10::IntArrayRef expect_size)
{
    bool is_inplace = check_inplace_tensor(src_list, dst);
    // Preserve legacy resizing behavior of out=... arguments
    if (!dst.sizes().equals(expect_size)) {
        TORCH_CHECK(!is_inplace, "output with shape ", dst.sizes(), " doesn't match the broadcast shape ", expect_size);
        dst.resize_(expect_size);
    }
    return;
}

constexpr aclDataType kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
#define DEFINE_ENUM(_1, n) n,
    AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(DEFINE_ENUM)
#undef DEFINE_ENUM
};

inline aclDataType ConvertToAclDataType(const at::ScalarType& data_type)
{
    auto acl_dtype = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(data_type)];
    TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED, std::string(c10::toString(data_type)) + " has not been supported")
    return acl_dtype;
}

inline c10::SmallVector<int64_t, SIZE> array_to_small_vector(c10::IntArrayRef shape)
{
    c10::SmallVector<int64_t, SIZE> shape_small_vec;
    for (uint64_t i = 0; i < shape.size(); i++) {
        shape_small_vec.emplace_back(shape[i]);
    }
    return shape_small_vec;
}

inline c10::SmallVector<int64_t, SIZE> conv_transpose2d_npu_output_size(const at::Tensor& input,
    const at::Tensor& weight, const at::Tensor& bias, c10::IntArrayRef padding, c10::IntArrayRef output_padding,
    c10::IntArrayRef stride, c10::IntArrayRef dilation, int64_t groups)
{
    int64_t N = input.size(0);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t Co = weight.size(1) * groups;
    auto kernel_size = weight.sizes().slice(2);

    int64_t Ho = (H - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1;
    int64_t Wo = (W - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1;

    c10::SmallVector<int64_t, SIZE> outputSize = {N, Co, Ho, Wo};

    return outputSize;
}

inline std::pair<bool, at::ScalarType> trans_torch_type_to_scalar(const std::string& type)
{
    if (dTypeTransMap.find(type) != dTypeTransMap.end()) {
        return {true, dTypeTransMap[type]};
    }
    return {false, at::ScalarType::Byte};
}

inline tuple_vector softmax_cross_entropy_with_logits_impl_npu_output_size(const at::Tensor& self)
{
    c10::SmallVector<int64_t, SIZE> resultSize = array_to_small_vector(self.size(0));
    c10::SmallVector<int64_t, SIZE> backpropSize = array_to_small_vector(self.sizes());

    return std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>(resultSize, backpropSize);
}

inline c10::SmallVector<int64_t, N> convert_array_to_vector(c10::IntArrayRef intArray)
{
    c10::SmallVector<int64_t, N> intVec;
    for (uint64_t i = 0; i < intArray.size(); i++) {
        intVec.emplace_back(intArray[i]);
    }
    return intVec;
}

inline int64_t make_warp_dim(int64_t dim, int64_t dim_post_expr)
{
    if (dim_post_expr <= 0) {
        dim_post_expr = 1; // this will make range [-1, 0]
    }
    if (dim < 0) {
        dim += dim_post_expr;
    }
    return dim;
}

// This logic is specially made for stride_add, and will be removed in future version.
inline c10::SmallVector<int64_t, SIZE> infersize_stride_add(c10::IntArrayRef shape1_, c10::IntArrayRef shape2_)
{
    auto shape1 = array_to_small_vector(shape1_);
    auto shape2 = array_to_small_vector(shape2_);

    c10::SmallVector<int64_t, SIZE> output_shape;
    if (shape1.size() < shape2.size()) {
        c10::SmallVector<int64_t, SIZE> shapeTemp = shape1;
        shape1 = shape2;
        shape2 = shapeTemp;
    }

    uint64_t shape1_size = shape1.size();
    uint64_t shape2_size = shape2.size();
    for (uint64_t i = 0; i < shape1_size - shape2_size; i++) {
        shape2.insert(shape2.begin(), 1);
    }

    for (uint64_t i = 0; i < shape1_size; i++) {
        if (shape1[i] == 0 || shape2[i] == 0) {
            output_shape.emplace_back((int64_t)0);
        } else {
            output_shape.emplace_back((shape1[i] > shape2[i]) ? shape1[i] : shape2[i]);
        }
    }
    return output_shape;
}

inline c10::SmallVector<int64_t, SIZE> transpose_npu_output_size(const at::Tensor& self, c10::IntArrayRef perm)
{
    auto sizes = self.sizes();
    c10::SmallVector<int64_t, SIZE> shape;
    for (uint64_t i = 0; i < perm.size(); i++) {
        shape.emplace_back(sizes[perm[i]]);
    }

    return shape;
}

inline bool check_match(const at::Tensor& self)
{
    static auto op =
        c10::Dispatcher::singleton().findSchemaOrThrow("aten::check_match", "").typed<bool(const at::Tensor&)>();
    return op.call(self);
}

inline void format_fresh_view(at::Tensor& x, const at::Tensor& y)
{
    x.copy_(y);
}

inline bool is_npu(const at::Tensor& tensor)
{
#ifdef COMPILE_WITH_XLA
    return tensor.device().type() == at::kXLA;
#else
    return tensor.device().type() == at::kPrivateUse1;
#endif
}

#define TORCH_CHECK_NPU(tensor) TORCH_CHECK(is_npu(tensor), #tensor " must be NPU tensor")
#endif // CSRC_COMMON_H_
