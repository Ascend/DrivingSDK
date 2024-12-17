import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving.point
from mx_driving import bev_pool_v2
from mx_driving._C import npu_bev_pool_v2_backward


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


# pylint: disable=too-many-arguments,huawei-too-many-arguments
@golden_data_cache(__file__)
def golden_bev_pool_v2(
    depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths, b, d, h, w, c
):
    output = np.zeros((b, d, h, w, c), dtype=np.float32)
    depth = depth.flatten()
    feat = feat.reshape((-1, c))
    output = output.reshape((-1, c))
    for start, length in zip(interval_starts, interval_lengths):
        for i in range(length):
            output[ranks_bev[start]] += depth[ranks_depth[start + i]] * feat[ranks_feat[start + i]]
    output = output.reshape((b, d, h, w, c))
    output = np.transpose(output, (0, 4, 1, 2, 3))
    return output


# pylint: disable=too-many-arguments,huawei-too-many-arguments
@golden_data_cache(__file__)
def golden_bev_pool_v2_grad(
    grad_out, depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths, b, d, h, w, c
):
    grad_depth = np.zeros_like(depth).flatten()
    grad_feat = np.zeros_like(feat).reshape((-1, c))
    depth = depth.flatten()
    feat = feat.reshape((-1, c))
    grad_out = grad_out.reshape((-1, c))
    for start, length in zip(interval_starts, interval_lengths):
        for i in range(length):
            gd = np.dot(grad_out[ranks_bev[start + i]], feat[ranks_feat[start + i]])
            grad_depth[ranks_depth[start + i]] = gd
            grad_feat[ranks_feat[start + i]] += depth[ranks_depth[start + i]] * grad_out[ranks_bev[start + i]]
    grad_feat = grad_feat.reshape((b, 1, h, w, c))
    return grad_feat


# pylint: disable=too-many-return-values
@golden_data_cache(__file__)
def generate_bev_pool_data(B, D, H, W, C, N_RANKS):
    feat = np.random.rand(B, 1, H, W, C).astype(np.float32)
    depth = np.random.rand(B, 1, D, H, W).astype(np.float32)
    grad_out = np.random.rand(B, D, H, W, C).astype(np.float32)
    ranks_depth = np.sort(np.random.randint(0, B * D * H * W, (N_RANKS,)).astype(np.int32))
    ranks_feat = np.sort(np.random.randint(0, B * H * W, (N_RANKS,)).astype(np.int32))
    ranks_bev = np.sort(np.random.randint(0, B * D * H * W, (N_RANKS,)).astype(np.int32))
    bev_feat_shape = (B, D, H, W, C)
    return feat, depth, grad_out, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape


class TestBEVPoolV2(TestCase):
    seed = 1024
    np.random.seed(seed)

    def test_bev_pool_v2(self):
        shapes = [
            [1, 1, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 3],
            [3, 3, 15, 15, 17, 33],
            [1, 5, 128, 128, 31, 777],
            [32, 4, 128, 128, 64, 9999],
        ]
        for shape in shapes:
            B, D, H, W, C, N_RANKS = shape
            feat, depth, grad_out, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape = generate_bev_pool_data(
                B, D, H, W, C, N_RANKS
            )
            kept = np.ones(ranks_bev.shape[0], dtype=bool)
            kept[1:] = ranks_feat[1:] != ranks_feat[:-1]
            interval_starts = np.where(kept)[0].astype(np.int32)
            interval_lengths = np.zeros_like(interval_starts, dtype=np.int32)
            interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
            interval_lengths[-1] = ranks_feat.shape[0] - interval_starts[-1]

            feat_npu = torch.from_numpy(feat).npu()
            grad_out_npu = torch.from_numpy(grad_out).npu()
            depth_npu = torch.from_numpy(depth).npu()
            ranks_depth_npu = torch.from_numpy(ranks_depth).npu()
            ranks_feat_npu = torch.from_numpy(ranks_feat).npu()
            ranks_bev_npu = torch.from_numpy(ranks_bev).npu()
            interval_lengths_npu = torch.from_numpy(interval_lengths).npu()
            interval_starts_npu = torch.from_numpy(interval_starts).npu()

            bev_feat = bev_pool_v2(
                depth_npu,
                feat_npu,
                ranks_depth_npu,
                ranks_feat_npu,
                ranks_bev_npu,
                (B, D, H, W, C),
                interval_starts_npu,
                interval_lengths_npu,
            )
            bev_feat_point = mx_driving.point.bev_pool_v2(
                depth_npu,
                feat_npu,
                ranks_depth_npu,
                ranks_feat_npu,
                ranks_bev_npu,
                (B, D, H, W, C),
                interval_starts_npu,
                interval_lengths_npu,
            )
            bev_feat_cpu = golden_bev_pool_v2(
                depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths, B, D, H, W, C
            )
            _, grad_feat_npu = npu_bev_pool_v2_backward(
                grad_out_npu,
                depth_npu,
                feat_npu,
                ranks_depth_npu,
                ranks_feat_npu,
                ranks_bev_npu,
                interval_lengths_npu,
                interval_starts_npu,
                B,
                D,
                H,
                W,
            )
            grad_feat = golden_bev_pool_v2_grad(
                grad_out,
                depth,
                feat,
                ranks_depth,
                ranks_feat,
                ranks_bev,
                interval_starts,
                interval_lengths,
                B,
                D,
                H,
                W,
                C,
            )
            self.assertRtolEqual(bev_feat.detach().cpu().numpy(), bev_feat_cpu)
            self.assertRtolEqual(grad_feat_npu.cpu().numpy(), grad_feat)


if __name__ == "__main__":
    run_tests()
