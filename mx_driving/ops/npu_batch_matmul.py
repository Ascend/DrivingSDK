"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""

import torch

def npu_batch_matmul(projection_mat, pts_extend):
    pts_extend = pts_extend.transpose(-1, -2).contiguous()
    broadcast_shape = [max(a, b) for a, b in zip(projection_mat.shape, pts_extend.shape)]
    projection_mat = projection_mat.expand(broadcast_shape).contiguous()
    pts_extend = pts_extend.expand(broadcast_shape).contiguous()
    dtpye = projection_mat.dtype
    if dtpye==torch.float16:
        result = torch.mul(projection_mat.float(), pts_extend.float())
        result = result.sum(dim=-1, keepdim=True)
        return result.to(dtpye)
    else:
        result = torch.mul(projection_mat, pts_extend)
        result = result.sum(dim=-1, keepdim=True)
        return result