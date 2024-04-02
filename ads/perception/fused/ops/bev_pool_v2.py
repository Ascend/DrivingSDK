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

import ads_c
import torch


class BEVPoolV2(torch.autograd.Function):
    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
        ranks_bev = ranks_bev.int()
        depth = depth.contiguous().float()
        feat = feat.contiguous().float()
        ranks_depth = ranks_depth.contiguous().int()
        ranks_feat = ranks_feat.contiguous().int()
        interval_lengths = interval_lengths.contiguous().int()
        interval_starts = interval_starts.contiguous().int()

        (B, D, H, W, C) = bev_feat_shape
        out = ads_c.npu_bev_pool_v2(
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W
        )

        ctx.save_for_backward(ranks_bev, depth, feat, ranks_feat, ranks_depth)
        ctx.saved_shapes = B, D, H, W
        return out

    @staticmethod
    # pylint: disable=too-many-return-values
    def backward(ctx, grad_out):
        ranks_bev, depth, feat, ranks_feat, ranks_depth = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes

        order = ranks_feat.argsort()
        ranks_feat, ranks_depth, ranks_bev = \
            ranks_feat[order], ranks_depth[order], ranks_bev[order]
        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_feat[1:] != ranks_feat[:-1]
        interval_starts_bp = torch.where(kept)[0].int()
        interval_lengths_bp = torch.zeros_like(interval_starts_bp)
        interval_lengths_bp[:-1] = interval_starts_bp[
                                   1:] - interval_starts_bp[:-1]
        interval_lengths_bp[-1] = ranks_bev.shape[0] - interval_starts_bp[-1]

        depth = depth.contiguous()
        feat = feat.contiguous()
        ranks_depth = ranks_depth.contiguous()
        ranks_feat = ranks_feat.contiguous()
        ranks_bev = ranks_bev.contiguous()
        interval_lengths_bp = interval_lengths_bp.contiguous()
        interval_starts_bp = interval_starts_bp.contiguous()
        grad_out = grad_out.contiguous()

        grad_depth, grad_feat = ads_c.npu_bev_pool_v2_backward(
            grad_out,
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths_bp,
            interval_starts_bp,
            B,
            D,
            H,
            W
        )
        return grad_depth, grad_feat, None, None, None, None, None, None


# pylint: disable=too-many-arguments,huawei-too-many-arguments
def bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
    x = BEVPoolV2.apply(
        depth, feat, ranks_depth, ranks_feat, ranks_bev,
        bev_feat_shape, interval_starts, interval_lengths
    )
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x
