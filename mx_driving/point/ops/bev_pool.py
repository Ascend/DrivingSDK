import torch

import mx_driving._C


class BEVPool(torch.autograd.Function):
    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx, feat, geom_feat, ranks, B, D, H, W):
        kept = torch.ones(feat.shape[0], device=feat.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts, dtype=torch.int32)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = feat.shape[0] - interval_starts[-1]
        geom_feat = geom_feat.int()

        out = mx_driving._C.npu_bev_pool(
            feat,
            geom_feat,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        ctx.save_for_backward(interval_starts, interval_lengths, geom_feat)
        ctx.saved_shapes = B, D, H, W
        return out

    @staticmethod
    # pylint: disable=too-many-return-values
    def backward(ctx, grad_out):
        interval_starts, interval_lengths, geom_feat = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes

        grad_out = grad_out.contiguous()
        grad_feat = mx_driving._C.npu_bev_pool_backward(
            grad_out,
            geom_feat,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        return grad_feat, None, None, None, None, None, None


# pylint: disable=too-many-arguments,huawei-too-many-arguments
def bev_pool(feat, geom_feat, B, D, H, W):
    """
    bev_pool is a function that pools the features in the BEV (Bird's Eye View) format.
    Please refer to the paper "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation"
    for more details.
    Args:
        feat (Tensor): The input feature tensor with shape (N, C).
        geom_feat (Tensor): The geometry feature tensor with shape (N, 4). The 4 elements are (h, w, d, b).
        B (int): The number of batch in the BEV.
        D (int): The number of depth in the BEV.
        H (int): The height of the BEV.
        W (int): The width of the BEV.
    Returns:
        bev_pooled_feat (Tensor): The pooled feature tensor with shape (B, C, D, H, W).
    Constraints:
        - The number of features and geometry features should be the same.
        - B * D * H * W * C <= 2^31, B, D <= 8, H, W <= 256, C <= 1024, for best practice.
        - C <= 1024
    Usage:
        >>> import torch, torch_npu
        >>> from mx_driving.perception.fused import bev_pool
        >>> feat = torch.rand(4, 256).npu()
        >>> feat.requires_grad_()
        >>> geom_feat = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3]], dtype=torch.int32).npu()
        >>> bev_pooled_feat = bev_pool(feat, geom_feat, 4, 1, 256, 256)
        >>> loss = bev_pooled_feat.sum()
        >>> loss.backward()
    """
    if feat.shape[0] != geom_feat.shape[0]:
        raise ValueError("The number of features and geometry features should be the same.")

    ranks = geom_feat[:, 0] * (W * D * B) + geom_feat[:, 1] * (D * B) + geom_feat[:, 2] * B + geom_feat[:, 3]
    indices = ranks.argsort()
    feat, geom_feat, ranks = feat[indices], geom_feat[indices], ranks[indices]

    out = BEVPool.apply(feat, geom_feat, ranks, B, D, H, W)
    out = out.permute(0, 4, 1, 2, 3).contiguous()
    return out
