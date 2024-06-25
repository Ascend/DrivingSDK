import unittest
import torch
import numpy as np
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from mx_driving.perception.fused import bev_pool

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


# pylint: disable=too-many-arguments,huawei-too-many-arguments
def golden_bev_pool(feat, geom_feat, b, d, h, w, c):
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


class TestBEVPool(TestCase):
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B',
                     "OP `bev_pool` is only supported on 910B, skip this ut!")
    def test_bev_pool(self):
        feat, geom_feat, out_shape = generate_bev_pool_data(1000, 64)
        (b, d, h, w, c) = out_shape
        feat_npu = torch.from_numpy(feat).npu()
        geom_feat_npu = torch.from_numpy(geom_feat).npu()
        out_npu = bev_pool(feat_npu, geom_feat_npu, b, d, h, w)
        out_cpu, interval_starts, interval_lengths = golden_bev_pool(feat, geom_feat, b, d, h, w, c)

        self.assertRtolEqual(out_cpu, out_npu.cpu().numpy())

if __name__ == '__main__':
    run_tests()
