// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.

#include "csrc/OpApiCommon.h"
#include "functions.h"

at::Tensor npu_hypot(const at::Tensor& x, const at::Tensor& y)
{
    auto out = at::empty_like(x, x.options());
    EXEC_NPU_CMD(aclnnHypot, x, y, out);
    return out;
}

std::tuple<at::Tensor, at::Tensor> npu_hypot_grad(const at::Tensor& x, const at::Tensor& y, const at::Tensor& out, const at::Tensor& out_grad)
{
    auto x_grad = at::empty_like(x, x.options());
    auto y_grad = at::empty_like(y, y.options());
    EXEC_NPU_CMD(aclnnHypotGrad, x, y, out, out_grad, x_grad, y_grad);
    return std::make_tuple(x_grad, y_grad);
}
