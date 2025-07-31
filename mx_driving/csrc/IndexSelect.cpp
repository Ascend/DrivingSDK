#include <iostream>

#include "csrc/OpApiCommon.h"
#include "csrc/functions.h"

namespace {
constexpr int8_t INPUT_DIM = 2;
constexpr int8_t INDEX_DIM = 1;
constexpr int8_t DIM_ZERO = 0;
constexpr int8_t FIRST_DIM = -2;
} // namespace

at::Tensor index_select(const at::Tensor& feature, int64_t dim, const at::Tensor& index)
{
    TORCH_CHECK_NPU(feature);
    TORCH_CHECK_NPU(index);
    TORCH_CHECK(feature.dim() == INPUT_DIM,
        "Feature must be a 2-D tensor, but received a tensor with dimension: ", feature.dim());
    TORCH_CHECK(dim == DIM_ZERO || dim == FIRST_DIM, "Dimension must be 0 or -2, but received: ", dim);
    TORCH_CHECK(
        index.dim() == INDEX_DIM, "Index must be a 1-D tensor, but received a tensor with dimension: ", index.dim());

    at::Tensor result = at::empty({index.size(0), feature.size(1)}, feature.options());

    EXEC_NPU_CMD(aclnnIndexSelect, feature, dim, index, result);

    return result;
}


at::Tensor index_select_backward(int64_t input_dim, int64_t dim, const at::Tensor& index, const at::Tensor& source)
{
    TORCH_CHECK_NPU(index);
    TORCH_CHECK_NPU(source);
    TORCH_CHECK(
        source.dim() == INPUT_DIM, "Source must be a 2-D tensor, but received a tensor with dimension: ", source.dim());
    TORCH_CHECK(dim == DIM_ZERO || dim == FIRST_DIM, "Dimension must be 0 or -2, but received: ", dim);
    TORCH_CHECK(
        index.dim() == INDEX_DIM, "Index must be a 1-D tensor, but received a tensor with dimension: ", index.dim());
    TORCH_CHECK(index.size(0) == source.size(0),
        "Index size must match the first dimension of source tensor, but received index size: ", index.size(0),
        " and source size: ", source.size(0));

    at::Tensor result = at::zeros({input_dim, source.size(1)}, source.options());

    int64_t mode = 0; // 高性能模式
    at::Scalar alpha = 1;
    EXEC_NPU_CMD(aclnnIndexAddV2, result, dim, index, source, alpha, mode, result);

    return result;
}
