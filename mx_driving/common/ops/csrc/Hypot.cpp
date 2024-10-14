// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.

#include "csrc/OpApiCommon.h"
#include "functions.h"

at::Tensor npu_hypot(const at::Tensor& input, const at::Tensor& other)
{
    auto out = at::empty_like(input, input.options());
    EXEC_NPU_CMD(aclnnHypot, input, other, out);
    return out;
}
