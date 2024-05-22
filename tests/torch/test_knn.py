import torch
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests
import ads.common


def reheap(dist, idx, k):
    root = 0
    child = root * 2 + 1
    while (child < k):
        if ((child + 1 < k) and (dist[child + 1] > dist[child])):
            child += 1
        if dist[root] > dist[child]:
            return

        dist[root], dist[child] = dist[child], dist[root]
        idx[root], idx[child] = idx[child], idx[root]
        root = child
        child = root * 2 + 1


def heap_sort(dist, idx, k):
    for i in range(k - 1, 0, -1):
        dist[0], dist[i] = dist[i], dist[0]
        idx[0], idx[i] = idx[i], idx[0]
        reheap(dist, idx, i)


class TestKnn(TestCase):
    def cpu_op_exec(self, arg1, source, target):
        using_gpu_alg, batch, npoint, N, nsample = arg1
        idx = np.zeros((batch, npoint, nsample), dtype=np.int32)
        dist2 = np.zeros((batch, npoint, nsample), dtype=np.float32)

        for b in range(batch):
            for m in range(npoint):
                new_x = target[b][m][0]
                new_y = target[b][m][1]
                new_z = target[b][m][2]

                x = source[b, :, 0]
                y = source[b, :, 1]
                z = source[b, :, 2]

                dist = (x - new_x) ** 2 + (y - new_y) ** 2 + (z - new_z) ** 2

                best_dist = np.ones((100,), dtype=np.float32) * 1e10
                best_idx = np.zeros((100,), dtype=np.int32)

                if using_gpu_alg:
                    for i in range(N):
                        if (dist[i] < best_dist[0]):
                            best_dist[0] = dist[i]
                            best_idx[0] = i
                            reheap(best_dist, best_idx, nsample)
                    heap_sort(best_dist, best_idx, nsample)
                    for i in range(nsample):
                        idx[b][m][i] = best_idx[i]
                        dist2[b][m][i] = best_dist[i]
                else:
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
        attrs = [False, b, m, n, k]

        expected_idx, _ = self.cpu_op_exec(arg1=attrs, source=xyz, target=center_xyz)
        idx = ads.common.knn(k, torch.from_numpy(xyz).npu(), torch.from_numpy(center_xyz).npu(), False)


if __name__ == "__main__":
    run_tests()