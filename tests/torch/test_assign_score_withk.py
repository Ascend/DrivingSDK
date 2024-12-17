import numpy as np
import torch
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving


# 'pylint: disable=too-many-arguments,huawei-too-many-arguments
@golden_data_cache(__file__)
def gen_data(B, N, npoint, M, K, out_dim):
    points = np.random.rand(B, N, M, out_dim).astype(np.float32)
    centers = np.random.rand(B, N, M, out_dim).astype(np.float32)
    scores = np.random.rand(B, npoint, K, M).astype(np.float32)
    knn_idx = np.random.randint(0, N, size=(B, npoint, K)).astype(np.int64)
    grad_out = np.random.rand(B, out_dim, npoint, K).astype(np.float32)
    data = [points, centers, scores, knn_idx, grad_out]
    return data


class TestAssignScoreWithk(TestCase):
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
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

    def test_assign_score_withk_should_return_right_value_when_shape_is_all_one(self):
        B = 1
        N = 1
        npoint = 1
        M = 1
        K = 1
        out_dim = 1
        points, centers, scores, knn_idx, _ = gen_data(B, N, npoint, M, K, out_dim)
        expected_output = self.cpu_forward_op(scores, points, centers, knn_idx, "sum")
        output = mx_driving.assign_score_withk(torch.from_numpy(scores).npu(),
                                                torch.from_numpy(points).npu(),
                                                torch.from_numpy(centers).npu(),
                                                torch.from_numpy(knn_idx).npu(),
                                                "sum")
        output = output.cpu().numpy()
        self.assertRtolEqual(expected_output, output)
    
    def test_assign_score_withk_should_return_right_value_when_shape_is_align_to_8(self):
        B = 8
        N = 32
        npoint = 8
        M = 8
        K = 8
        out_dim = 16
        points, centers, scores, knn_idx, _ = gen_data(B, N, npoint, M, K, out_dim)
        expected_output = self.cpu_forward_op(scores, points, centers, knn_idx, "sum")
        output = mx_driving.assign_score_withk(torch.from_numpy(scores).npu(),
                                                torch.from_numpy(points).npu(),
                                                torch.from_numpy(centers).npu(),
                                                torch.from_numpy(knn_idx).npu(),
                                                "sum")
        output = output.cpu().numpy()
        self.assertRtolEqual(expected_output, output)

    def test_assign_score_withk_should_return_right_value_when_shape_is_not_align(self):
        B = 21
        N = 43
        npoint = 31
        M = 9
        K = 25
        out_dim = 11
        points, centers, scores, knn_idx, _ = gen_data(B, N, npoint, M, K, out_dim)
        expected_output = self.cpu_forward_op(scores, points, centers, knn_idx, "sum")
        output = mx_driving.assign_score_withk(torch.from_numpy(scores).npu(),
                                                torch.from_numpy(points).npu(),
                                                torch.from_numpy(centers).npu(),
                                                torch.from_numpy(knn_idx).npu(),
                                                "sum")
        output = output.cpu().numpy()
        self.assertRtolEqual(expected_output, output)

    def test_assign_score_withk_should_return_right_value_when_M_is_1000(self):
        B = 3
        N = 32
        npoint = 10
        M = 1000
        K = 8
        out_dim = 11
        points, centers, scores, knn_idx, _ = gen_data(B, N, npoint, M, K, out_dim)
        expected_output = self.cpu_forward_op(scores, points, centers, knn_idx, "sum")
        output = mx_driving.assign_score_withk(torch.from_numpy(scores).npu(),
                                                torch.from_numpy(points).npu(),
                                                torch.from_numpy(centers).npu(),
                                                torch.from_numpy(knn_idx).npu(),
                                                "sum")
        output = output.cpu().numpy()
        self.assertRtolEqual(expected_output, output)

    def test_assign_score_withk_should_raise_error_value_when_npoint_is_larger_than_N(self):
        B = 3
        N = 23
        npoint = 24
        M = 8
        K = 8
        out_dim = 11
        points, centers, scores, knn_idx, _ = gen_data(B, N, npoint, M, K, out_dim)
        points.reshape(N, B, M * out_dim)
        try:
            output = mx_driving.assign_score_withk(torch.from_numpy(scores).npu(),
                                                    torch.from_numpy(points).npu(),
                                                    torch.from_numpy(centers).npu(),
                                                    torch.from_numpy(knn_idx).npu(),
                                                    "sum")
        except Exception as e:
            assert "The number of whole points must be larger than or equal to the number of sample points." in str(e)

    def test_assign_score_withk_should_raise_error_value_when_K_is_larger_than_N(self):
        B = 3
        N = 23
        npoint = 20
        M = 8
        K = 24
        out_dim = 11
        points, centers, scores, knn_idx, _ = gen_data(B, N, npoint, M, K, out_dim)
        points.reshape(N, B, M * out_dim)
        try:
            output = mx_driving.assign_score_withk(torch.from_numpy(scores).npu(),
                                                    torch.from_numpy(points).npu(),
                                                    torch.from_numpy(centers).npu(),
                                                    torch.from_numpy(knn_idx).npu(),
                                                    "sum")
        except Exception as e:
            assert "The number of whole points must be larger than or equal to the number of neighbors." in str(e)
    
    def test_assign_score_withk_should_raise_error_value_when_npoint_is_zero(self):
        B = 3
        N = 23
        npoint = 0
        M = 8
        K = 24
        out_dim = 11
        points, centers, scores, knn_idx, _ = gen_data(B, N, npoint, M, K, out_dim)
        points.reshape(N, B, M * out_dim)
        try:
            output = mx_driving.assign_score_withk(torch.from_numpy(scores).npu(),
                                                    torch.from_numpy(points).npu(),
                                                    torch.from_numpy(centers).npu(),
                                                    torch.from_numpy(knn_idx).npu(),
                                                    "sum")
        except Exception as e:
            assert "Error! Input shape can not contain zero!" in str(e)


class TestAssignScoreWithkGrad(TestCase):
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    @golden_data_cache(__file__)
    def cpu_backward_op(self,
        grad_out,
        scores,
        points,
        centers,
        knn_idx,
        aggregate):
        agg = {'sum': 0, 'avg': 1, 'max': 2}

        B, N, M, out_dim = points.shape
        _, npoint, K, _ = scores.shape

        points_grad = np.zeros([B, N, M, out_dim], dtype=points.dtype)
        centers_grad = np.zeros([B, N, M, out_dim], dtype=centers.dtype)
        scores_grad = np.zeros([B, npoint, K, M], dtype=scores.dtype)

        for b in range(B):
            for n in range(npoint):
                for k in range(K):
                    scores_tmp = np.tile(scores[b, n, k].reshape(M, 1), (1, out_dim))
                    gradout_tmp = np.tile(grad_out[b, :, n, k], (M, 1))
                    points_grad[b, knn_idx[b, n, k]] += scores_tmp * gradout_tmp
                    centers_grad[b, knn_idx[b, n, 0]] -= scores_tmp * gradout_tmp

                    temp1 = points[b, knn_idx[b, n, k]] - centers[b, knn_idx[b, n, 0]]
                    temp2 = np.tile(grad_out[b, :, n, k], (M, 1))
                    scores_grad[b, n, k] = np.sum(temp1 * temp2, axis=1)
        return scores_grad, points_grad, centers_grad
    
    def test_assign_score_withk_grad_should_return_right_value_when_shape_is_all_one(self):
        B = 1
        N = 1
        npoint = 1
        M = 1
        K = 1
        out_dim = 1
        points, centers, scores, knn_idx, grad_out = gen_data(B, N, npoint, M, K, out_dim)
        points_npu = torch.from_numpy(points).npu()
        centers_npu = torch.from_numpy(centers).npu()
        scores_npu = torch.from_numpy(scores).npu()
        knn_idx_npu = torch.from_numpy(knn_idx).npu()
        grad_out_npu = torch.from_numpy(grad_out).npu()
        points_npu.requires_grad = True
        centers_npu.requires_grad = True
        scores_npu.requires_grad = True
        output = mx_driving.assign_score_withk(scores_npu, points_npu, centers_npu, knn_idx_npu, "sum")
        output.backward(grad_out_npu)
        expected_output = self.cpu_backward_op(grad_out, scores, points, centers, knn_idx, "sum")
        self.assertRtolEqual(expected_output[0], scores_npu.grad.detach().cpu().numpy())
        self.assertRtolEqual(expected_output[1], points_npu.grad.detach().cpu().numpy())
        self.assertRtolEqual(expected_output[2], centers_npu.grad.detach().cpu().numpy())

    def test_assign_score_withk_grad_should_return_right_value_when_shape_is_align_to_8(self):
        B = 8
        N = 32
        npoint = 8
        M = 8
        K = 8
        out_dim = 16
        points, centers, scores, knn_idx, grad_out = gen_data(B, N, npoint, M, K, out_dim)
        points_npu = torch.from_numpy(points).npu()
        centers_npu = torch.from_numpy(centers).npu()
        scores_npu = torch.from_numpy(scores).npu()
        knn_idx_npu = torch.from_numpy(knn_idx).npu()
        grad_out_npu = torch.from_numpy(grad_out).npu()
        points_npu.requires_grad = True
        centers_npu.requires_grad = True
        scores_npu.requires_grad = True
        output = mx_driving.assign_score_withk(scores_npu, points_npu, centers_npu, knn_idx_npu, "sum")
        output.backward(grad_out_npu)
        expected_output = self.cpu_backward_op(grad_out, scores, points, centers, knn_idx, "sum")
        self.assertRtolEqual(expected_output[0], scores_npu.grad.detach().cpu().numpy())
        self.assertRtolEqual(expected_output[1], points_npu.grad.detach().cpu().numpy())
        self.assertRtolEqual(expected_output[2], centers_npu.grad.detach().cpu().numpy())

    def test_assign_score_withk_grad_should_return_right_value_when_shape_is_not_align(self):
        B = 21
        N = 43
        npoint = 31
        M = 9
        K = 25
        out_dim = 11
        points, centers, scores, knn_idx, grad_out = gen_data(B, N, npoint, M, K, out_dim)
        points_npu = torch.from_numpy(points).npu()
        centers_npu = torch.from_numpy(centers).npu()
        scores_npu = torch.from_numpy(scores).npu()
        knn_idx_npu = torch.from_numpy(knn_idx).npu()
        grad_out_npu = torch.from_numpy(grad_out).npu()
        points_npu.requires_grad = True
        centers_npu.requires_grad = True
        scores_npu.requires_grad = True
        output = mx_driving.assign_score_withk(scores_npu, points_npu, centers_npu, knn_idx_npu, "sum")
        output.backward(grad_out_npu)
        expected_output = self.cpu_backward_op(grad_out, scores, points, centers, knn_idx, "sum")
        self.assertRtolEqual(expected_output[0], scores_npu.grad.detach().cpu().numpy())
        self.assertRtolEqual(expected_output[1], points_npu.grad.detach().cpu().numpy())
        self.assertRtolEqual(expected_output[2], centers_npu.grad.detach().cpu().numpy())

    def test_assign_score_withk_should_return_right_value_when_M_plus_out_dim_is_equals_to_5000(self):
        B = 21
        N = 43
        npoint = 31
        M = 10
        K = 25
        out_dim = 500
        points, centers, scores, knn_idx, grad_out = gen_data(B, N, npoint, M, K, out_dim)
        points_npu = torch.from_numpy(points).npu()
        centers_npu = torch.from_numpy(centers).npu()
        scores_npu = torch.from_numpy(scores).npu()
        knn_idx_npu = torch.from_numpy(knn_idx).npu()
        grad_out_npu = torch.from_numpy(grad_out).npu()
        points_npu.requires_grad = True
        centers_npu.requires_grad = True
        scores_npu.requires_grad = True
        output = mx_driving.assign_score_withk(scores_npu, points_npu, centers_npu, knn_idx_npu, "sum")
        output.backward(grad_out_npu)
        expected_output = self.cpu_backward_op(grad_out, scores, points, centers, knn_idx, "sum")
        self.assertRtolEqual(expected_output[0], scores_npu.grad.detach().cpu().numpy())
        self.assertRtolEqual(expected_output[1], points_npu.grad.detach().cpu().numpy())
        self.assertRtolEqual(expected_output[2], centers_npu.grad.detach().cpu().numpy())
    
    def test_assign_score_withk_should_raise_error_when_M_plus_out_dim_is_equals_5025(self):
        B = 21
        N = 43
        npoint = 31
        M = 10
        K = 25
        out_dim = 501
        points, centers, scores, knn_idx, grad_out = gen_data(B, N, npoint, M, K, out_dim)
        points_npu = torch.from_numpy(points).npu()
        centers_npu = torch.from_numpy(centers).npu()
        scores_npu = torch.from_numpy(scores).npu()
        knn_idx_npu = torch.from_numpy(knn_idx).npu()
        grad_out_npu = torch.from_numpy(grad_out).npu()
        points_npu.requires_grad = True
        centers_npu.requires_grad = True
        scores_npu.requires_grad = True
        output = mx_driving.assign_score_withk(scores_npu, points_npu, centers_npu, knn_idx_npu, "sum")
        try:
            output.backward(grad_out_npu)
            raise Exception("The value of M * out_dim is larger than 5000")
        except Exception as e:
            assert "The size of M or K is too large" in str(e)


if __name__ == "__main__":
    run_tests()
    