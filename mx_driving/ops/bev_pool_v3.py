# Copyright (c) 2024 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Phigent Robotics. All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Union

import torch

import mx_driving._C


class BEVPoolV3(torch.autograd.Function):
    """
    BEVPoolV3 adapts BEVPoolV1 and BEVPoolV2 for best performance on NPU.
    """

    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(
        ctx,
        depth: Union[torch.Tensor, None],
        feat: torch.Tensor,
        ranks_depth: Union[torch.Tensor, None],
        ranks_feat: Union[torch.Tensor, None],
        ranks_bev: torch.Tensor,
        bev_feat_shape: List[int],
    ) -> torch.Tensor:
        (B, D, H, W, C) = bev_feat_shape
        feat = feat.contiguous()
        if depth is None:
            if ranks_bev.dim() != 2:
                raise ValueError("ranks_bev must be 2D when running without depth")
            ranks_bev = ranks_bev[:, 3] * D * H * W + ranks_bev[:, 2] * H * W + ranks_bev[:, 0] * W + ranks_bev[:, 1]
        out = mx_driving._C.npu_bev_pool_v3(depth, feat, ranks_depth, ranks_feat, ranks_bev, B, D, H, W)
        ctx.save_for_backward(depth, feat, ranks_feat, ranks_depth, ranks_bev)
        return out

    @staticmethod
    # pylint: disable=too-many-return-values
    def backward(ctx, grad_out: torch.Tensor):
        depth, feat, ranks_feat, ranks_depth, ranks_bev = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_depth, grad_feat = mx_driving._C.npu_bev_pool_v3_backward(
            grad_out,
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
        )
        return grad_depth, grad_feat, None, None, None, None


# pylint: disable=too-many-arguments,huawei-too-many-arguments
def bev_pool_v3(
    depth: Union[torch.Tensor, None],
    feat: torch.Tensor,
    ranks_depth: Union[torch.Tensor, None],
    ranks_feat: Union[torch.Tensor, None],
    ranks_bev: torch.Tensor,
    bev_feat_shape: List[int],
) -> torch.Tensor:
    x = BEVPoolV3.apply(
        depth,
        feat,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        bev_feat_shape,
    )
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x
