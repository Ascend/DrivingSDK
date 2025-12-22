// Copyright (c) 2025 Huawei Technologies Co., Ltd

#include "csrc/OpApiCommon.h"
#include "csrc/functions.h"

namespace {
constexpr int64_t MAX_INCHANNEL = 1024;
const int64_t DIVISOR = 8;
const size_t INITIAL_CAPACITY = 2;
} // namespace

at::Tensor npu_sparse_inverse_conv3d(const at::Tensor& features, const at::Tensor& origin_indices,
    const at::Tensor& unique_indices_offset, const at::Tensor& sorted_idx_to_former_indices,
    at::IntArrayRef kernel_size, int in_channel)
{
    TORCH_CHECK_NPU(features);
    TORCH_CHECK_NPU(origin_indices);
    TORCH_CHECK_NPU(unique_indices_offset);
    TORCH_CHECK_NPU(sorted_idx_to_former_indices);
    TORCH_CHECK(in_channel <= MAX_INCHANNEL,
        "in_channel must less or equal than 1024 expected but got in_channel: ", in_channel);
    TORCH_CHECK(in_channel % DIVISOR == 0, "in_channel must be divisible by 8 but got in_channel: ", in_channel);

    int64_t kernel_sum = 1;
    for (int32_t i = 0; i < static_cast<int32_t>(kernel_size.size()); i++) {
        kernel_sum *= kernel_size[i];
    }

    c10::SmallVector<int64_t, INITIAL_CAPACITY> output_size = {origin_indices.sizes()[0], kernel_sum * in_channel};
    at::Tensor output_img2col = at::zeros(output_size, features.options());
    EXEC_NPU_CMD(aclnnSparseInverseConv3d, features, origin_indices, unique_indices_offset,
        sorted_idx_to_former_indices, kernel_size, in_channel, output_img2col);

    return output_img2col;
}
