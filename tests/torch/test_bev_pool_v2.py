import unittest
import torch
import numpy as np
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from ads.perception.fused import bev_pool_v2

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


# pylint: disable=too-many-arguments,huawei-too-many-arguments
def golden_bev_pool_v2(feat, geom_feat, b, d, h, w, c):
    output = np.zeros((b, d, h, w, c), dtype=np.float32)
    ranks = geom_feat[:, 0] * (w * d * b) + geom_feat[:, 1] * (d * b) + geom_feat[:, 2] * b + geom_feat[:, 3]
    indices = np.argsort(ranks)
    feat, geom_feat, ranks = feat[indices], geom_feat[indices], ranks[indices]
    kept = np.ones(feat.shape[0], dtype=bool)
    kept[1:] = ranks[1:] != ranks[:-1]
    interval_starts = np.where(kept)[0].astype(np.int32)
    interval_lengths = np.zeros_like(interval_starts, dtype=np.int32)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = feat.shape[0] - interval_starts[-1]
    for (start, length) in zip(interval_starts, interval_lengths):
        geom = geom_feat[start]
        for i in range(length):
            output[geom[3], geom[2], geom[0], geom[1], :] += feat[start + i, :]
    output = np.transpose(output, (0, 4, 1, 2, 3))
    return output, interval_starts, interval_lengths


# pylint: disable=too-many-arguments,huawei-too-many-arguments
def golden_bev_pool_grad(feat, geom_feat, interval_starts, interval_lengths, grad_output, b, d, h, w, c):
    grad_feat = np.zeros_like(feat)
    for (start, length) in zip(interval_starts, interval_lengths):
        geom = geom_feat[start]
        for i in range(length):
            grad_feat[start + i, :] = grad_output[geom[3], geom[2], geom[0], geom[1], :]

    return grad_feat


def generate_bev_pool_data(n, c):
    feat = np.random.rand(n, c).astype(np.float32)
    geom_feat = np.random.randint(0, 32, (n, 4)).astype(np.int32)
    out_shape = (32, 32, 32, 32, c)
    return feat, geom_feat, out_shape


class TestBEVPoolV2(TestCase):
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B',
                     "OP `bev_pool` is only supported on 910B, skip this ut!")
    def test_bev_pool_v2(self):
        depth = np.array([0.3, 0.4, 0.2, 0.1, 0.7, 0.6, 0.8, 0.9])
        depth = torch.from_numpy(depth).float().npu()
        depth = depth.view(1, 1, 2, 2, 2).requires_grad_()
        feat = torch.ones(
            size=[1, 1, 2, 2, 2], dtype=torch.float).npu()
        feat.requires_grad_()
        ranks_depth = torch.from_numpy(np.array([0, 4, 1, 6])).int().npu()
        ranks_feat = torch.from_numpy(np.array([0, 0, 1, 2])).int().npu()
        ranks_bev = torch.from_numpy(np.array([0, 0, 1, 1])).int().npu()

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               (1, 1, 2, 2, 2), interval_starts, interval_lengths)
        loss = torch.sum(bev_feat)
        loss.backward()
        grad_depth = np.array([2., 2., 0., 0., 2., 0., 2., 0.])
        grad_depth = torch.from_numpy(grad_depth).float()
        grad_depth = grad_depth.npu().view(1, 1, 2, 2, 2)
        self.assertRtolEqual(depth.grad.cpu().numpy(), grad_depth.cpu().numpy())
        grad_feat = np.array([1.0, 1.0, 0.4, 0.4, 0.8, 0.8, 0., 0.])
        grad_feat = torch.from_numpy(grad_feat).float().npu().view(1, 1, 2, 2, 2)
        self.assertRtolEqual(feat.grad.cpu().numpy(), grad_feat.cpu().numpy())


if __name__ == '__main__':
    run_tests()
