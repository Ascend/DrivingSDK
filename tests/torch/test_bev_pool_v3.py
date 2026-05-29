import torch
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

from mx_driving import bev_pool_v3
from mx_driving.ops.bev_pool_v3 import BEVPoolV3


class TestBEVPoolV3(TestCase):
    seed = 1024

    def setUp(self):
        torch.manual_seed(self.seed)

        class MockCtx:
            def __init__(self, saved_tensors):
                self.saved_tensors = saved_tensors

        self.MockCtx = MockCtx
        self.shapes = [
            [1, 1, 1, 1, 8, 1],
            [3, 3, 3, 3, 16, 3],
            [3, 3, 15, 15, 32, 33],
            [1, 5, 17, 23, 8, 777],
            [32, 7, 11, 17, 64, 9999],
        ]

    @staticmethod
    def ranks_bev_2d_to_1d(ranks_bev, D, H, W):
        # 列对应: [:, 0]=H, [:, 1]=W, [:, 2]=D, [:, 3]=B
        return (
            ranks_bev[:, 3].to(torch.int32) * (D * H * W)
            + ranks_bev[:, 2].to(torch.int32) * (H * W)
            + ranks_bev[:, 0].to(torch.int32) * W
            + ranks_bev[:, 1].to(torch.int32)
        )

    @golden_data_cache(__file__)
    def golden_bev_pool_v3(self, depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape):
        B, D, H, W, C = bev_feat_shape
        feat_flat = feat.view(-1, C)
        out_flat = torch.zeros(B * D * H * W, C, dtype=feat.dtype, device=feat.device)

        if depth is not None:
            depth_flat = depth.view(-1)
            d_vals = depth_flat[ranks_depth]  # [N_RANKS]
            f_vals = feat_flat[ranks_feat]  # [N_RANKS, C]
            weighted = d_vals.unsqueeze(1) * f_vals
            out_flat.index_add_(0, ranks_bev, weighted)
        else:
            if ranks_bev.dim() != 2:
                raise ValueError("ranks_bev must be 2D when running without depth")
            linear_idx = self.ranks_bev_2d_to_1d(ranks_bev, D, H, W).long()
            out_flat.index_add_(0, linear_idx, feat_flat)

        out = out_flat.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        return out

    @golden_data_cache(__file__)
    def golden_bev_pool_v3_grad_with_depth(self, bev_feat_cpu, grad_out, feat, depth):
        bev_feat_cpu.backward(grad_out)
        return feat.grad, depth.grad

    @golden_data_cache(__file__)
    def golden_bev_pool_v3_grad_without_depth(self, bev_feat_cpu, grad_out, feat):
        bev_feat_cpu.backward(grad_out)
        return feat.grad

    def generate_bev_pool_data(self, input_shape, with_depth=True):
        B, D, H, W, C, N_RANKS = input_shape
        grad_out = torch.rand([B, C, D, H, W]) * 10 - 5
        bev_feat_shape = [B, D, H, W, C]

        if with_depth:
            depth = torch.rand([B, 1, D, H, W]) * 10 - 5
            feat = torch.rand([B, 1, H, W, C]) * 10 - 5
            ranks_depth = torch.randint(0, B * D * H * W, [N_RANKS], dtype=torch.int32)
            ranks_feat = torch.randint(0, B * H * W, [N_RANKS], dtype=torch.int32)
            ranks_bev = torch.randint(0, B * D * H * W, [N_RANKS], dtype=torch.int32)
            return feat, depth, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape, grad_out
        else:
            feat = torch.rand([N_RANKS, C]) * 10 - 5
            ranks_bev = torch.stack(
                [
                    torch.randint(0, H, (N_RANKS,)),  # [:, 0] 范围 [0, H-1]
                    torch.randint(0, W, (N_RANKS,)),  # [:, 1] 范围 [0, W-1]
                    torch.randint(0, D, (N_RANKS,)),  # [:, 2] 范围 [0, D-1]
                    torch.randint(0, B, (N_RANKS,)),  # [:, 3] 范围 [0, B-1]
                ],
                dim=1,
            ).to(torch.int32)
            return feat, None, None, None, ranks_bev, bev_feat_shape, grad_out

    def test_bev_pool_v3_with_depth(self):
        for shape in self.shapes:
            feat, depth, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape, _ = self.generate_bev_pool_data(
                input_shape=shape, with_depth=True
            )
            bev_feat_cpu = self.golden_bev_pool_v3(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape)
            bev_feat_npu = bev_pool_v3(
                depth.npu(), feat.npu(), ranks_depth.npu(), ranks_feat.npu(), ranks_bev.npu(), bev_feat_shape
            )
            self.assertRtolEqual(bev_feat_npu.detach().cpu(), bev_feat_cpu)

    def test_bev_pool_v3_grad_with_depth(self):
        for shape in self.shapes:
            feat, depth, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape, grad_out = self.generate_bev_pool_data(
                input_shape=shape, with_depth=True
            )
            feat_npu = feat.clone().npu()
            depth_npu = depth.clone().npu()

            feat.requires_grad_()
            depth.requires_grad_()
            bev_feat_cpu = self.golden_bev_pool_v3(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape)
            grad_feat_cpu, grad_depth_cpu = self.golden_bev_pool_v3_grad_with_depth(bev_feat_cpu, grad_out, feat, depth)

            saved_tensors = (depth_npu, feat_npu, ranks_feat.npu(), ranks_depth.npu(), ranks_bev.npu())
            grad_depth_npu, grad_feat_npu, _, _, _, _ = BEVPoolV3.backward(self.MockCtx(saved_tensors), grad_out.npu())

            self.assertRtolEqual(grad_feat_npu.detach().cpu(), grad_feat_cpu)
            self.assertRtolEqual(grad_depth_npu.detach().cpu(), grad_depth_cpu)

    def test_bev_pool_v3_without_depth(self):
        for shape in self.shapes:
            feat, _, _, _, ranks_bev, bev_feat_shape, _ = self.generate_bev_pool_data(
                input_shape=shape, with_depth=False
            )
            bev_feat_cpu = self.golden_bev_pool_v3(None, feat, None, None, ranks_bev, bev_feat_shape)
            bev_feat_npu = bev_pool_v3(None, feat.npu(), None, None, ranks_bev.npu(), bev_feat_shape)
            self.assertRtolEqual(bev_feat_npu.detach().cpu(), bev_feat_cpu)

    def test_bev_pool_v3_grad_without_depth(self):
        for shape in self.shapes:
            feat, _, _, _, ranks_bev, bev_feat_shape, grad_out = self.generate_bev_pool_data(
                input_shape=shape, with_depth=False
            )
            feat_npu = feat.clone().npu()

            feat.requires_grad_()
            bev_feat_cpu = self.golden_bev_pool_v3(None, feat, None, None, ranks_bev, bev_feat_shape)
            grad_feat_cpu = self.golden_bev_pool_v3_grad_without_depth(bev_feat_cpu, grad_out, feat)

            _, D, H, W, _ = bev_feat_shape
            ranks_bev_1d = self.ranks_bev_2d_to_1d(ranks_bev, D, H, W)
            saved_tensors = (None, feat_npu, None, None, ranks_bev_1d.npu())
            _, grad_feat_npu, _, _, _, _ = BEVPoolV3.backward(self.MockCtx(saved_tensors), grad_out.npu())

            self.assertRtolEqual(grad_feat_npu.detach().cpu(), grad_feat_cpu)


if __name__ == "__main__":
    run_tests()
