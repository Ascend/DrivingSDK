import unittest

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving.point
from mx_driving import bev_pool


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


# pylint: disable=too-many-arguments,huawei-too-many-arguments
@golden_data_cache(__file__)
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
    for start, length in zip(interval_starts, interval_lengths):
        geom = geom_feat[start]
        for i in range(length):
            output[geom[3], geom[2], geom[0], geom[1], :] += feat[start + i, :]
    output = np.transpose(output, (0, 4, 1, 2, 3))
    return output, interval_starts, interval_lengths


# pylint: disable=too-many-arguments,huawei-too-many-arguments
@golden_data_cache(__file__)
def golden_bev_pool_grad(feat, geom_feat, interval_starts, interval_lengths, grad_output, b, d, h, w, c):
    grad_feat = np.zeros_like(feat)
    for start, length in zip(interval_starts, interval_lengths):
        geom = geom_feat[start]
        for i in range(length):
            grad_feat[start + i, :] = grad_output[geom[3], geom[2], geom[0], geom[1], :]

    return grad_feat


@golden_data_cache(__file__)
def generate_bev_pool_data(n, b, d, h, w, c):
    feat = np.random.rand(n, c).astype(np.float32)
    geom_feat_b = np.random.randint(0, b, (n,)).astype(np.int32)
    geom_feat_d = np.random.randint(0, d, (n,)).astype(np.int32)
    geom_feat_h = np.random.randint(0, h, (n,)).astype(np.int32)
    geom_feat_w = np.random.randint(0, w, (n,)).astype(np.int32)
    geom_feat = np.stack([geom_feat_h, geom_feat_w, geom_feat_d, geom_feat_b], axis=1)
    return feat, geom_feat


class TestBEVPool(TestCase):
    seed = 1024
    np.random.seed(seed)

    def test_bev_pool(self):
        shapes = [
            [1, 1, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 3],
            [3, 3, 15, 15, 17, 33],
            [1, 5, 128, 128, 31, 777],
            [32, 4, 128, 128, 64, 9999],
        ]
        for shape in shapes:
            (b, d, h, w, c, n) = shape
            feat, geom_feat = generate_bev_pool_data(n, b, d, h, w, c)
            feat_npu = torch.from_numpy(feat).npu()
            geom_feat_npu = torch.from_numpy(geom_feat).npu()
            out_npu = bev_pool(feat_npu, geom_feat_npu, b, d, h, w)
            out_point = mx_driving.point.bev_pool(feat_npu, geom_feat_npu, b, d, h, w)
            out_cpu, interval_starts, interval_lengths = golden_bev_pool(feat, geom_feat, b, d, h, w, c)

            self.assertRtolEqual(out_cpu, out_npu.cpu().numpy())


if __name__ == "__main__":
    run_tests()
