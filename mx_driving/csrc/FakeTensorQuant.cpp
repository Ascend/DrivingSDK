/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */


#include "csrc/OpApiCommon.h"
#include "csrc/functions.h"

at::Tensor npu_fake_tensor_quant(const at::Tensor& inputs, const at::Tensor& amax, const int num_bits,
    const bool is_unsigned, const bool narrow_range)
{
    TORCH_CHECK_NPU(inputs);
    TORCH_CHECK_NPU(amax);
    at::Tensor out = at::empty_like(inputs);
    EXEC_NPU_CMD(aclnnFakeTensorQuant, inputs, amax, num_bits, is_unsigned, narrow_range, out);
    return out;
}

at::Tensor npu_fake_tensor_quant_inplace(const at::Tensor& inputs, const at::Tensor& amax, const int num_bits,
    const bool is_unsigned, const bool narrow_range)
{
    TORCH_CHECK_NPU(inputs);
    TORCH_CHECK_NPU(amax);
    at::Tensor out = inputs;
    EXEC_NPU_CMD(aclnnFakeTensorQuant, inputs, amax, num_bits, is_unsigned, narrow_range, out);
    return out;
}

at::Tensor npu_fake_tensor_quant_with_axis(const at::Tensor& inputs, const at::Tensor& amax, const int axis,
    const int num_bits, const bool is_unsigned, const bool narrow_range)
{
    TORCH_CHECK_NPU(inputs);
    TORCH_CHECK_NPU(amax);
    at::Tensor out = at::empty_like(inputs);
    EXEC_NPU_CMD(aclnnFakeTensorQuantWithAxis, inputs, amax, axis, num_bits, is_unsigned, narrow_range, out);
    return out;
}
