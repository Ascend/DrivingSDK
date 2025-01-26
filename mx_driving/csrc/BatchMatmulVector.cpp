// Copyright (c) 2024 Huawei Technologies Co., Ltd

#include "csrc/OpApiCommon.h"
#include "csrc/functions.h"

at::Tensor npu_batch_matmul(const at::Tensor& projection_mat, const at::Tensor& pts_extend)
{
    auto result = at::empty_like(pts_extend, pts_extend.options());
    EXEC_NPU_CMD(aclnnBatchMatmulVector, projection_mat, pts_extend, result);
    return result;
}
