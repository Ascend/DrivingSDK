import unittest

import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

from mx_driving import bev_pool_v3


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


# pylint: disable=too-many-arguments,huawei-too-many-arguments
@golden_data_cache(__file__)
def golden_bev_pool_v3(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape):
    B, D, H, W, C = bev_feat_shape
    N_RANKS = ranks_depth.shape[0]
    depth = depth.view([-1])
    feat = feat.view([-1, C])
    
    out = torch.zeros([B * D * H * W, C])
    for i in range(N_RANKS):
        d = depth[ranks_depth[i]]
        f = feat[ranks_feat[i]]
        b = ranks_bev[i]
        out[b] += d * f
    out = out.view(bev_feat_shape)
    
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


class TestBEVPoolV2(TestCase):
    seed = 1024
    torch.manual_seed(seed)

    def test_bev_pool_v2(self):
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
            bev_feat_npu.backward(grad_out_npu)

            self.assertRtolEqual(bev_feat_npu.detach().cpu().numpy(), bev_feat_cpu.detach().cpu().numpy())
            self.assertRtolEqual(feat_npu.grad.cpu().numpy(), bev_feat_grad_cpu.cpu().numpy())
            self.assertRtolEqual(depth_npu.grad.cpu().numpy(), bev_depth_grad_cpu.cpu().numpy())


if __name__ == "__main__":
    run_tests()
