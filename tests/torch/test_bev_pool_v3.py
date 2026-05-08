import unittest

import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

from mx_driving import bev_pool_v3


# pylint: disable=too-many-arguments,huawei-too-many-arguments
@golden_data_cache(__file__)
def golden_bev_pool_v3(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape):
    B, D, H, W, C = bev_feat_shape
    depth = depth.view(-1)
    feat = feat.view(-1, C)

    d_vals = depth[ranks_depth]  # [N_RANKS]
    f_vals = feat[ranks_feat]    # [N_RANKS, C]
    weighted = d_vals.unsqueeze(1) * f_vals

    # 使用 index_add_ 实现高效累加（支持 GPU）
    out = torch.zeros(B * D * H * W, C, device=depth.device)
    out.index_add_(0, ranks_bev, weighted)  # 在第0维上按索引累加

    out = out.view(bev_feat_shape)
    # 调整维度顺序：[B, D, H, W, C] -> [B, C, D, H, W]
    out = torch.permute(out, [0, 4, 1, 2, 3])

    return out


@golden_data_cache(__file__)
def golden_bev_pool_v3_grad(bev_feat_cpu, grad_out, feat, depth):
    bev_feat_cpu.backward(grad_out)
    
    return feat.grad, depth.grad


# pylint: disable=too-many-return-values
@golden_data_cache(__file__)
def generate_bev_pool_data(B, D, H, W, C, N_RANKS):
    depth = torch.rand([B, 1, D, H, W])
    feat = torch.rand([B, 1, H, W, C])
    ranks_depth = torch.randint(0, B * D * H * W, [N_RANKS], dtype=torch.int32)
    ranks_feat = torch.randint(0, B * H * W, [N_RANKS], dtype=torch.int32)
    ranks_bev = torch.randint(0, B, [N_RANKS], dtype=torch.int32)
    grad_out = torch.rand([B, C, D, H, W])
    bev_feat_shape = [B, D, H, W, C]
    return feat, depth, grad_out, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape


def golden_bev_pool_v3_no_depth(feat, ranks_feat, ranks_bev_2d, bev_feat_shape):
    """
    无 depth 时的 golden 函数
    - feat: [B, C, H, W] 输入特征
    - ranks_feat: 未使用 (NPU侧未使用此参数)
    - ranks_bev_2d: [N_RANKS, 4] 格式为 [H, W, D, B]
    """
    B, D, H, W, C = bev_feat_shape

    # 从 feat 中按顺序提取特征 (与 NPU 逻辑一致: taskIdx * avgRankNum * channel_)
    feat_flat = feat.view(-1, C)  # [B*H*W, C]

    # 将 2D ranks_bev 转换为 1D 线性索引
    # 列对应: [:, 0]=H, [:, 1]=W, [:, 2]=D, [:, 3]=B
    h_idx = ranks_bev_2d[:, 0].long()  # H
    w_idx = ranks_bev_2d[:, 1].long()  # W
    d_idx = ranks_bev_2d[:, 2].long()  # D
    b_idx = ranks_bev_2d[:, 3].long()  # B

    # 转换为 1D 索引 (B, D, H, W) -> 1D
    linear_idx = b_idx * (D * H * W) + d_idx * (H * W) + h_idx * W + w_idx

    # 创建输出 [B*D*H*W, C]
    out_flat = torch.zeros(B * D * H * W, C, dtype=feat.dtype, device=feat.device)

    # scatter_add: 将 valid_feats 添加到 linear_idx 位置
    out_flat.scatter_add_(0, linear_idx.unsqueeze(1).expand(-1, C), feat_flat)

    # reshape 回 [B, D, H, W, C] -> [B, C, D, H, W]
    out = out_flat.view(B, D, H, W, C)
    out = out.permute(0, 4, 1, 2, 3).contiguous()
    return out


class TestBEVPoolV3(TestCase):
    seed = 1024
    torch.manual_seed(seed)

    def test_bev_pool_v3(self):
        class MockCtx:
            def __init__(self, saved_tensors):
                self.saved_tensors = saved_tensors
        shapes = [
            [1, 1, 1, 1, 8, 1],
            [3, 3, 3, 3, 16, 3],
            [3, 3, 15, 15, 32, 33],
            [1, 5, 17, 23, 8, 777],
            [32, 7, 11, 17, 64, 9999],
        ]
        for shape in shapes:
            B, D, H, W, C, N_RANKS = shape
            feat, depth, grad_out, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape = generate_bev_pool_data(
                B, D, H, W, C, N_RANKS
            )
            
            feat_npu = feat.clone().to("npu")
            depth_npu = depth.clone().to("npu")
            grad_out_npu = grad_out.clone().to("npu")
            ranks_depth_npu = ranks_depth.clone().to("npu")
            ranks_feat_npu = ranks_feat.clone().to("npu")
            ranks_bev_npu = ranks_bev.clone().to("npu")
            depth.requires_grad_()
            feat.requires_grad_()
            feat_npu.requires_grad_()
            depth_npu.requires_grad_()

            bev_feat_cpu = golden_bev_pool_v3(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape)
            bev_feat_grad_cpu, bev_depth_grad_cpu = golden_bev_pool_v3_grad(bev_feat_cpu, grad_out, feat, depth)

            bev_feat_npu = bev_pool_v3(
                depth_npu, feat_npu, ranks_depth_npu, ranks_feat_npu, ranks_bev_npu, bev_feat_shape
            )
            saved_tensors = (depth_npu, feat_npu, ranks_feat_npu, ranks_depth_npu, ranks_bev_npu)
            ctx = MockCtx(saved_tensors)
            from mx_driving.ops.bev_pool_v3 import BEVPoolV3
            grad_depth_npu, grad_feat_npu, _, _, _, _ = BEVPoolV3.backward(ctx, grad_out_npu.permute(0, 2, 3, 4, 1).contiguous())
            self.assertRtolEqual(bev_feat_npu.detach().cpu().numpy(), bev_feat_cpu.detach().cpu().numpy())
            self.assertRtolEqual(grad_feat_npu.cpu().numpy(), bev_feat_grad_cpu.cpu().numpy())
            self.assertRtolEqual(grad_depth_npu.cpu().numpy(), bev_depth_grad_cpu.cpu().numpy())

    def test_none_valid_ranks_bev(self):
        B, D, H, W, C, N_RANKS = 1, 1, 1, 1, 8, 1
        feat, _, grad_out, _, ranks_feat, ranks_bev, bev_feat_shape = generate_bev_pool_data(
                B, D, H, W, C, N_RANKS
            )
        depth = None
        ranks_depth = None
        ranks_bev_2d = torch.tensor([
            [torch.randint(0, B, ()).item(),    # batch_idx: [0, B-1]
            torch.randint(0, W, ()).item(),    # width_idx: [0, W-1]
            torch.randint(0, H, ()).item(),    # height_idx: [0, H-1]
            torch.randint(0, D, ()).item()]    # depth_idx: [0, D-1]
            for _ in range(N_RANKS)
        ], dtype=torch.int32).to("npu")

        feat_npu = feat.clone().to("npu").requires_grad_(True)
        grad_out_npu = grad_out.clone().to("npu")
        ranks_feat_npu = ranks_feat.clone().to("npu")
        ranks_bev_npu = ranks_bev.clone().to("npu")

        with self.assertRaises(Exception) as ctx:
            bev_feat_npu = bev_pool_v3(depth, feat_npu, ranks_depth, ranks_feat_npu, ranks_bev_npu, bev_feat_shape)
        self.assertEqual(str(ctx.exception), "ranks_bev must be 2D when running without depth")


    def test_depth_none_valid_ranks_bev(self):
        shapes = [
            [1, 1, 1, 1, 8, 1],
            [3, 3, 3, 3, 16, 3],
            [3, 3, 15, 15, 32, 33],
            [1, 5, 17, 23, 8, 777],
            [32, 7, 11, 17, 64, 9999],
        ]
        for shape in shapes:
            B, D, H, W, C, N_RANKS = shape
            bev_feat_shape = [B, D, H, W, C]

            feat_npu = torch.rand(B, C, H, W, D).to("npu")

            # ranks_feat 在无 depth 模式下 NPU 侧未使用，传 None 即可
            ranks_feat_npu = None
            ranks_feat_cpu = None

            # 列对应关系：[:, 3]=B, [:, 2]=D, [:, 0]=H, [:, 1]=W
            ranks_bev_2d_cpu = torch.stack([
                torch.randint(0, H, (N_RANKS,)),  # [:, 0] 范围 [0, H-1]
                torch.randint(0, W, (N_RANKS,)),  # [:, 1] 范围 [0, W-1]
                torch.randint(0, D, (N_RANKS,)),  # [:, 2] 范围 [0, D-1]
                torch.randint(0, B, (N_RANKS,))   # [:, 3] 范围 [0, B-1]
            ], dim=1).to(torch.int32)

            ranks_bev_2d_npu = ranks_bev_2d_cpu.clone().to("npu")

            depth = None
            ranks_depth = None

            out_npu = bev_pool_v3(depth, feat_npu, ranks_depth, ranks_feat_npu, ranks_bev_2d_npu, bev_feat_shape)
            out_cpu = golden_bev_pool_v3_no_depth(feat_npu.cpu(), ranks_feat_cpu, ranks_bev_2d_cpu, bev_feat_shape)
            self.assertRtolEqual(out_cpu.detach().numpy(), out_npu.detach().cpu().numpy())


if __name__ == "__main__":
    run_tests()