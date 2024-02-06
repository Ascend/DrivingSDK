#ifndef __COMMON_H__
#define __COMMON_H__
#include <ATen/ATen.h>
#include <string>
#include <tuple>
#include <vector>
#include <ATen/core/dispatch/Dispatcher.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/framework/utils/NPUDefinition.h"
#include "third_party/acl/inc/acl/acl_base.h"

const int N = 32;
const int SIZE = 8;

using tuple_vector = std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>;
aclDataType ConvertToAclDataType(const at::ScalarType &data_type);
c10::SmallVector<int64_t, SIZE> array_to_small_vector(c10::IntArrayRef shape);
c10::SmallVector<int64_t, SIZE> conv_transpose2d_npu_output_size(const at::Tensor &input, const at::Tensor &weight,
                                                                 const at::Tensor &bias, c10::IntArrayRef padding,
                                                                 c10::IntArrayRef output_padding,
                                                                 c10::IntArrayRef stride, c10::IntArrayRef dilation,
                                                                 int64_t groups);

std::pair<bool, at::ScalarType> trans_torch_type_to_scalar(const std::string &type);
tuple_vector softmax_cross_entropy_with_logits_impl_npu_output_size(const at::Tensor& self);
int64_t make_warp_dim(int64_t dim, int64_t dim_post_expr);
c10::SmallVector<int64_t, N> convert_array_to_vector(c10::IntArrayRef intArray);
c10::SmallVector<int64_t, SIZE> infersize_stride_add(c10::IntArrayRef shape1_, c10::IntArrayRef shape2_);
c10::SmallVector<int64_t, SIZE> transpose_npu_output_size(const at::Tensor &self, c10::IntArrayRef perm);
bool check_match(const at::Tensor &self);
void format_fresh_view(at::Tensor &x, const at::Tensor &y);

#endif // __COMMON_H__
