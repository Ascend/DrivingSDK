import torch
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests
import ads.common


class TestThreeNN(TestCase):
    def cpu_op_exec(self):
        idx = np.zeros((self.batch, self.npoint, 3), dtype=np.int32)
        dist2 = np.zeros((self.batch, self.npoint, 3), dtype=np.float32)

        for b in range(self.batch):
            for m in range(self.npoint):
                new_x = self.target[b][m][0]
                new_y = self.target[b][m][1]
                new_z = self.target[b][m][2]

                x = self.source[b, :, 0]
                y = self.source[b, :, 1]
                z = self.source[b, :, 2]

                dist = (x - new_x) ** 2 + (y - new_y) ** 2 + (z - new_z) ** 2

                sorted_indices_and_values = sorted(enumerate(dist), key=lambda x: (x[1], x[0]))
                for i in range(3):
                    idx[b][m][i], dist2[b][m][i] = sorted_indices_and_values[i]
        return np.sqrt(dist2), idx

    def test_three_nn(self):
        self.batch = 1
        self.npoint = 1
        self.N = 200
        self.source = np.ones((self.batch, self.N, 3)).astype(np.float32)
        self.target = np.zeros((self.batch, self.npoint, 3)).astype(np.float32)

        expected_dist, expected_idx = self.cpu_op_exec()
        dist, idx = ads.common.three_nn(torch.from_numpy(self.target).npu(), torch.from_numpy(self.source).npu())

        self.assertRtolEqual(expected_dist, dist.cpu().numpy())
        self.assertRtolEqual(expected_idx, idx.cpu().numpy())


if __name__ == "__main__":
    run_tests()