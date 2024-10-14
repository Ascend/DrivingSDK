import torch
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests
import mx_driving.common


class TestAssignScoreWithk(TestCase):
    def cpu_forward_op(self,
        scores,
        points,
        centers,
        knn_idx,
        aggregate):
        agg = {"sum": 0, "avg": 1, "max": 2}
        B, N, M, out_dim = points.shape
        _, npoint, K, _ = scores.shape
        output = np.zeros([B, npoint, K, out_dim], dtype=points.dtype)
        for b in range(B):
            for n in range(npoint):
                output_tmp = np.zeros([K, out_dim], dtype=points.dtype)
                for m in range(M):
                    p = points[b, knn_idx[b, n, :], m, :]
                    c = centers[b, knn_idx[b, n, 0], m, :]
                    s = scores[b, n, :, m]
                    tmp = np.zeros([K, out_dim], dtype=points.dtype)
                    for k in range(K):
                        tmp[k] = (p[k] - c) * s[k]
                    output_tmp += tmp
                output[b, n] = output_tmp
        return output.transpose(0, 3, 1, 2)
    
    def gen_data(self, attrs):
        B, N, npoint, M, K, out_dim = attrs
        points = np.random.rand(B, N, M, out_dim).astype(np.float32)
        centers = np.random.rand(B, N, M, out_dim).astype(np.float32)
        scores = np.random.rand(B, npoint, K, M).astype(np.float32)
        knn_idx = np.random.randint(0, N, size=(B, npoint, K)).astype(np.int32)
        data = [points, centers, scores, knn_idx]
        return data

    def test_assign_score_withk_case_1(self):
        B = 1
        N = 1
        npoint = 1
        M = 6
        K = 1
        out_dim = 1
        attrs = [B, N, npoint, M, K, out_dim]
        points, centers, scores, knn_idx = self.gen_data(attrs)
        expected_output = self.cpu_forward_op(scores, points, centers, knn_idx, "sum")
        output = mx_driving.common.assign_score_withk(torch.from_numpy(scores).npu(),
                                                        torch.from_numpy(points).npu(),
                                                        torch.from_numpy(centers).npu(),
                                                        torch.from_numpy(knn_idx).npu(),
                                                        "sum")
        self.assertRtolEqual(expected_output, output.cpu().numpy())
    
    def test_assign_score_withk_case_2(self):
        B = 32
        N = 512
        npoint = 256
        M = 8
        K = 32
        out_dim = 32
        attrs = [B, N, npoint, M, K, out_dim]
        points, centers, scores, knn_idx = self.gen_data(attrs)
        expected_output = self.cpu_forward_op(scores, points, centers, knn_idx, "sum")
        output = mx_driving.common.assign_score_withk(torch.from_numpy(scores).npu(),
                                                        torch.from_numpy(points).npu(),
                                                        torch.from_numpy(centers).npu(),
                                                        torch.from_numpy(knn_idx).npu(),
                                                        "sum")
        self.assertRtolEqual(expected_output, output.cpu().numpy())

    

if __name__ == "__main__":
    run_tests()