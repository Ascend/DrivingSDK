import numpy as np
import torch
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving
import mx_driving.common


@golden_data_cache(__file__)
def cpu_gen_inputs(attrs):
    batch, npoint, N, nsample, transposed = attrs
    idx = np.zeros((batch, npoint, nsample), dtype=np.int32)
    dist2 = np.zeros((batch, npoint, nsample), dtype=np.float32)

    return idx, dist2


class TestKnn(TestCase):
    @golden_data_cache(__file__)
    def cpu_op_exec(self,
        attrs,
        xyz,
        center_xyz):
        batch, npoint, N, nsample, transposed = attrs
        idx, dist2 = cpu_gen_inputs(attrs)
        if transposed:
            xyz = np.transpose(xyz, axes=(0, 2, 1))
            center_xyz = np.transpose(center_xyz, axes=(0, 2, 1))
        for b in range(batch):
            for m in range(npoint):
                new_x = center_xyz[b][m][0]
                new_y = center_xyz[b][m][1]
                new_z = center_xyz[b][m][2]

                x = xyz[b, :, 0]
                y = xyz[b, :, 1]
                z = xyz[b, :, 2]

                dist = (x - new_x) ** 2 + (y - new_y) ** 2 + (z - new_z) ** 2

                indices_to_replace = np.where(dist > 1e10)
                dist[indices_to_replace] = 1e10
                sorted_indices_and_values = sorted(enumerate(dist), key=lambda x: (x[1], x[0]))
                for i in range(nsample):
                    idx[b][m][i], dist2[b][m][i] = sorted_indices_and_values[i]
                    if i >= N - len(indices_to_replace[0]):
                        idx[b][m][i] = 0
        return np.transpose(idx, axes=(0, 2, 1)), dist2

    def test_knn(self):
        b = 1
        m = 1
        n = 200
        k = 3
        xyz = np.ones((b, n, 3)).astype(np.float32)
        center_xyz = np.zeros((b, m, 3)).astype(np.float32)

        expected_idx, _ = self.cpu_op_exec([b, m, n, k, False], xyz, center_xyz)
        idx = mx_driving.knn(k, torch.from_numpy(xyz).npu(), torch.from_numpy(center_xyz).npu(), False)
        self.assertRtolEqual(expected_idx, idx.cpu().numpy())

        idx_verify = mx_driving.common.knn(k, torch.from_numpy(xyz).npu(), torch.from_numpy(center_xyz).npu(), False)
        self.assertRtolEqual(expected_idx, idx_verify.cpu().numpy())

    def test_knn_1(self):
        b = 30
        m = 256
        n = 1024
        k = 3
        xyz = np.ones((b, n, 3)).astype(np.float32)
        center_xyz = np.zeros((b, m, 3)).astype(np.float32)

        expected_idx, _ = self.cpu_op_exec([b, m, n, k, False], xyz, center_xyz)
        idx = mx_driving.knn(k, torch.from_numpy(xyz).npu(), torch.from_numpy(center_xyz).npu(), False)
        self.assertRtolEqual(expected_idx, idx.cpu().numpy())

        idx_verify = mx_driving.common.knn(k, torch.from_numpy(xyz).npu(), torch.from_numpy(center_xyz).npu(), False)
        self.assertRtolEqual(expected_idx, idx_verify.cpu().numpy())

    def test_knn_case2(self):
        b = 8
        m = 1024
        n = 4096
        k = 32
        np.random.seed(0)
        xyz = np.random.randn(b, n, 3).astype(np.float32)
        center_xyz = np.random.randn(b, m, 3).astype(np.float32)

        expected_idx, _ = self.cpu_op_exec([b, m, n, k, False], xyz, center_xyz)
        idx = mx_driving.knn(k, torch.from_numpy(xyz).npu(), torch.from_numpy(center_xyz).npu(), False)
        self.assertRtolEqual(expected_idx, idx.cpu().numpy())

        idx_verify = mx_driving.common.knn(k, torch.from_numpy(xyz).npu(), torch.from_numpy(center_xyz).npu(), False)
        self.assertRtolEqual(expected_idx, idx_verify.cpu().numpy())

    def test_knn_case3(self):
        b = 32
        m = 4096
        n = 4096
        k = 8
        np.random.seed(0)
        xyz = np.random.randn(b, n, 3).astype(np.float32)
        center_xyz = np.random.randn(b, m, 3).astype(np.float32)

        expected_idx, _ = self.cpu_op_exec([b, m, n, k, False], xyz, center_xyz)
        idx = mx_driving.knn(k, torch.from_numpy(xyz).npu(), torch.from_numpy(center_xyz).npu(), False)
        self.assertRtolEqual(expected_idx, idx.cpu().numpy())

        idx_verify = mx_driving.common.knn(k, torch.from_numpy(xyz).npu(), torch.from_numpy(center_xyz).npu(), False)
        self.assertRtolEqual(expected_idx, idx_verify.cpu().numpy())

    def test_knn_case4(self):
        b = 8
        m = 256
        n = 1024
        k = 8
        np.random.seed(0)
        xyz = np.random.randn(b, n, 3).astype(np.float32)
        center_xyz = np.random.randn(b, m, 3).astype(np.float32)

        expected_idx, _ = self.cpu_op_exec([b, m, n, k, False], xyz, center_xyz)
        idx = mx_driving.knn(k, torch.from_numpy(xyz).npu(), torch.from_numpy(center_xyz).npu(), False)
        self.assertRtolEqual(expected_idx, idx.cpu().numpy())

        idx_verify = mx_driving.common.knn(k, torch.from_numpy(xyz).npu(), torch.from_numpy(center_xyz).npu(), False)
        self.assertRtolEqual(expected_idx, idx_verify.cpu().numpy())

    def test_knn_case5(self):
        b = 8
        m = 64
        n = 256
        k = 8
        np.random.seed(0)
        xyz = np.random.randn(b, n, 3).astype(np.float32)
        center_xyz = np.random.randn(b, m, 3).astype(np.float32)

        expected_idx, _ = self.cpu_op_exec([b, m, n, k, False], xyz, center_xyz)
        idx = mx_driving.knn(k, torch.from_numpy(xyz).npu(), torch.from_numpy(center_xyz).npu(), False)
        self.assertRtolEqual(expected_idx, idx.cpu().numpy())

        idx_verify = mx_driving.common.knn(k, torch.from_numpy(xyz).npu(), torch.from_numpy(center_xyz).npu(), False)
        self.assertRtolEqual(expected_idx, idx_verify.cpu().numpy())

    def test_knn_case6(self):
        b = 8
        m = 16
        n = 64
        k = 8
        np.random.seed(0)
        xyz = np.random.randn(b, n, 3).astype(np.float32)
        center_xyz = np.random.randn(b, m, 3).astype(np.float32)

        expected_idx, _ = self.cpu_op_exec([b, m, n, k, False], xyz, center_xyz)
        idx = mx_driving.knn(k, torch.from_numpy(xyz).npu(), torch.from_numpy(center_xyz).npu(), False)
        self.assertRtolEqual(expected_idx, idx.cpu().numpy())

        idx_verify = mx_driving.common.knn(k, torch.from_numpy(xyz).npu(), torch.from_numpy(center_xyz).npu(), False)
        self.assertRtolEqual(expected_idx, idx_verify.cpu().numpy())


if __name__ == "__main__":
    run_tests()