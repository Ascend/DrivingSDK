import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving


@golden_data_cache(__file__)
def gen_data(bs, k1, k2, dim):
    pt = np.random.rand(bs, k1, dim).astype(np.float32)
    poly_line = np.random.rand(bs, k2, dim).astype(np.float32)
    data = [pt, poly_line]
    return data


@golden_data_cache(__file__)
def gen_data_multi_d(n, bs, k1, k2, dim):
    pt = np.random.rand(n, bs, k1, dim).astype(np.float32)
    poly_line = np.random.rand(n, bs, k2, dim).astype(np.float32)
    data = [pt, poly_line]
    return data


class TestCartesianToFrenet(TestCase):
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def batch_index_select(self, A, indices):
        """ Slice input tensor with indicies in batch-wise
        Args:
            A: (B x A x N x D) or (B x N x D)
            indices: (B x A x M) or (B x M) (select M indices, element value between 0~N-1)
        return:
            out: (B x A x M x D) or (B x M x D)
        """
        assert isinstance(A, torch.Tensor) or isinstance(A, np.ndarray), "Input type must be np.ndarray or torch.Tensor"
        assert len(A.shape) == 3 or len(A.shape) == 4, "Input shape must be with 3 or 4 dimensions"
        if len(A.shape) == 3:
            assert len(indices.shape) == 2 and indices.shape[:1] == A.shape[:1], f"Input size mismatched"
            b, n, d = A.shape
            if isinstance(A, torch.Tensor):
                indicies_offset = indices.type(torch.int32) + torch.arange(0, b * n, n, dtype=torch.int32, device=indices.device).unsqueeze(1)
                output = A.reshape(b * n, d)[indicies_offset.to(dtype=torch.int64)]
            elif isinstance(A, np.ndarray):
                indicies_offset = indices + np.arange(0, b * n, n)[..., np.newaxis]
                output = A.reshape(b * n, d)[indicies_offset]
            return output
        if len(A.shape) == 4:
            assert len(indices.shape) == 3 and indices.shape[:2] == A.shape[:2], f"Input size mismatched"
            b, a, n, d = A.shape
            _, _, m = indices.shape
            indices = indices.reshape(b * a, m)
            indicies_offset = indices.type(torch.int32) + torch.arange(0, b * a * n, n, dtype=torch.int32, device=indices.device).unsqueeze(1)
            output = A.reshape(b * a * n, d)[indicies_offset.to(dtype=torch.int64)]
            return output.reshape(b, a, m, d)
        return None

    def golden_op(self, pt, poly_line):
        """
        Args:
            pt: (*shape, k1, 2)
            poly_line: (*shape, k2, 2)
            *shape must can be broadcasted
        Returns:
            pt_sl: (*shape, k1, 2)
        """

        def _dot(pts1, pts2):
            return torch.sum((pts1 * pts2), -1)

        def _cross(pts1, pts2):
            return pts1[..., 0] * pts2[..., 1] - pts1[..., 1] * pts2[..., 0]

        def _pt_to_line(pt, pt1, pt2):
            sub_pt_pt1 = pt - pt1
            sub_pt2_pt1 = pt2 - pt1
            s = _dot(sub_pt_pt1, sub_pt2_pt1) / torch.clip(torch.norm(sub_pt2_pt1, dim=-1), 1e-5, 1e10)
            length = -_cross(sub_pt_pt1, sub_pt2_pt1) / torch.clip(torch.norm(sub_pt2_pt1, dim=-1), 1e-5, 1e10)
            return s, length

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1
        diff_tensor = torch.zeros(poly_line_shape, device=poly_line.device)
        diff_tensor = torch.cat([diff_tensor, poly_line[..., 1:, :] - poly_line[..., :-1, :]], dim=-2)
        diff_dist = torch.norm(diff_tensor, dim=-1)  # (*shape, k2)
        s_cum = torch.cumsum(diff_dist, dim=-1)  # (*shape, k2)

        dist_vec = torch.unsqueeze(pt, dim=-2) - torch.unsqueeze(poly_line, dim=-3)  # [s, p, t, 2]
        dist_all = torch.norm(dist_vec, dim=-1)  # (*shape, k1, k2) [s, p, t]
        min_idx = torch.argmin(dist_all, dim=-1).int()  # [s, p]
        front_point = self.batch_index_select(poly_line, min_idx)
        zero = torch.tensor(0, dtype=min_idx.dtype, device=min_idx.device)
        one = torch.tensor(1, dtype=min_idx.dtype, device=min_idx.device)

        back_index = torch.where(min_idx - one < zero, zero, min_idx - one)
        back_point = self.batch_index_select(poly_line, back_index)
        move_forward_point = (_dot(pt - front_point, back_point - front_point) > zero)
        tmp_idx = torch.logical_and(min_idx > zero,
                                    move_forward_point)
        tmp = torch.tensor(poly_line.shape[-2] - 1, dtype=min_idx.dtype, device=min_idx.device)
        tmp_idx = torch.logical_or(tmp_idx, min_idx == tmp)
        min_idx = torch.where(tmp_idx, min_idx - one, min_idx)

        poly_start = self.batch_index_select(poly_line, min_idx)
        poly_end = self.batch_index_select(poly_line, min_idx + 1)
        s, length = _pt_to_line(pt, poly_start, poly_end)  # (*shape, k1)
        s = s + self.batch_index_select(s_cum.unsqueeze(-1), min_idx).squeeze(-1)  # (*shape, k1)
        return poly_start, poly_end, torch.stack((s, length), dim=-1)  # (*shape, k1, 2)




    def test_cartesian_to_frenet_should_return_right_value_when_shape_is_all_one(self):
        bs = 1
        k1 = 1
        k2 = 2
        dim = 2

        pt, poly_line = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        poly_start_expected, poly_end_expected, sl_expected = self.golden_op(pt, poly_line)
        poly_start, poly_end, sl = mx_driving.cartesian_to_frenet(pt, poly_line)

        poly_start_expected = poly_start_expected.cpu().numpy()
        poly_end_expected = poly_end_expected.cpu().numpy()
        sl_expected = sl_expected.cpu().numpy()

        poly_start = poly_start.cpu().numpy()
        poly_end = poly_end.cpu().numpy()
        sl = sl.cpu().numpy()

        self.assertRtolEqual(poly_start_expected, poly_start)
        self.assertRtolEqual(poly_end_expected, poly_end)
        self.assertRtolEqual(sl_expected, sl)

    def test_cartesian_to_frenet_should_return_right_value_when_shape_is_align_to_8(self):
        bs = 32
        k1 = 8
        k2 = 8
        dim = 2

        pt, poly_line = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        poly_start_expected, poly_end_expected, sl_expected = self.golden_op(pt, poly_line)
        poly_start, poly_end, sl = mx_driving.cartesian_to_frenet(pt, poly_line)

        poly_start_expected = poly_start_expected.cpu().numpy()
        poly_end_expected = poly_end_expected.cpu().numpy()
        sl_expected = sl_expected.cpu().numpy()

        poly_start = poly_start.cpu().numpy()
        poly_end = poly_end.cpu().numpy()
        sl = sl.cpu().numpy()

        self.assertRtolEqual(poly_start_expected, poly_start)
        self.assertRtolEqual(poly_end_expected, poly_end)
        self.assertRtolEqual(sl_expected, sl)

    def test_cartesian_to_frenet_should_return_right_value_when_shape_is_not_align(self):
        bs = 110
        k1 = 30
        k2 = 15
        dim = 2

        pt, poly_line = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        poly_start_expected, poly_end_expected, sl_expected = self.golden_op(pt, poly_line)
        poly_start, poly_end, sl = mx_driving.cartesian_to_frenet(pt, poly_line)

        poly_start_expected = poly_start_expected.cpu().numpy()
        poly_end_expected = poly_end_expected.cpu().numpy()
        sl_expected = sl_expected.cpu().numpy()

        poly_start = poly_start.cpu().numpy()
        poly_end = poly_end.cpu().numpy()
        sl = sl.cpu().numpy()

        self.assertRtolEqual(poly_start_expected, poly_start)
        self.assertRtolEqual(poly_end_expected, poly_end)
        self.assertRtolEqual(sl_expected, sl)

    def test_cartesian_to_frenet_should_return_right_value_when_bs_is_5000(self):
        bs = 5000
        k1 = 40
        k2 = 20
        dim = 2

        pt, poly_line = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        poly_start_expected, poly_end_expected, sl_expected = self.golden_op(pt, poly_line)
        poly_start, poly_end, sl = mx_driving.cartesian_to_frenet(pt, poly_line)

        poly_start_expected = poly_start_expected.cpu().numpy()
        poly_end_expected = poly_end_expected.cpu().numpy()
        sl_expected = sl_expected.cpu().numpy()

        poly_start = poly_start.cpu().numpy()
        poly_end = poly_end.cpu().numpy()
        sl = sl.cpu().numpy()

        self.assertRtolEqual(poly_start_expected, poly_start)
        self.assertRtolEqual(poly_end_expected, poly_end)
        self.assertRtolEqual(sl_expected, sl)

    def test_cartesian_to_frenet_should_return_right_value_when_bs_greater_than_5000(self):
        bs = 10000
        k1 = 40
        k2 = 20
        dim = 2

        pt, poly_line = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        poly_start_expected, poly_end_expected, sl_expected = self.golden_op(pt, poly_line)
        poly_start, poly_end, sl = mx_driving.cartesian_to_frenet(pt, poly_line)

        poly_start_expected = poly_start_expected.cpu().numpy()
        poly_end_expected = poly_end_expected.cpu().numpy()
        sl_expected = sl_expected.cpu().numpy()

        poly_start = poly_start.cpu().numpy()
        poly_end = poly_end.cpu().numpy()
        sl = sl.cpu().numpy()

        self.assertRtolEqual(poly_start_expected, poly_start)
        self.assertRtolEqual(poly_end_expected, poly_end)
        self.assertRtolEqual(sl_expected, sl)

    def test_cartesian_to_frenet_should_return_right_value_when_shape_maxed(self):
        bs = 10000
        k1 = 256
        k2 = 256
        dim = 2

        pt, poly_line = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        poly_start_expected, poly_end_expected, sl_expected = self.golden_op(pt, poly_line)
        poly_start, poly_end, sl = mx_driving.cartesian_to_frenet(pt, poly_line)

        poly_start_expected = poly_start_expected.cpu().numpy()
        poly_end_expected = poly_end_expected.cpu().numpy()
        sl_expected = sl_expected.cpu().numpy()

        poly_start = poly_start.cpu().numpy()
        poly_end = poly_end.cpu().numpy()
        sl = sl.cpu().numpy()

        self.assertRtolEqual(poly_start_expected, poly_start)
        self.assertRtolEqual(poly_end_expected, poly_end)
        self.assertRtolEqual(sl_expected, sl)

    def test_cartesian_to_frenet_should_return_right_value_when_k2_is_larger_than_k1(self):
        bs = 5000
        k1 = 20
        k2 = 40
        dim = 2

        pt, poly_line = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        poly_start_expected, poly_end_expected, sl_expected = self.golden_op(pt, poly_line)
        poly_start, poly_end, sl = mx_driving.cartesian_to_frenet(pt, poly_line)

        poly_start_expected = poly_start_expected.cpu().numpy()
        poly_end_expected = poly_end_expected.cpu().numpy()
        sl_expected = sl_expected.cpu().numpy()

        poly_start = poly_start.cpu().numpy()
        poly_end = poly_end.cpu().numpy()
        sl = sl.cpu().numpy()

        self.assertRtolEqual(poly_start_expected, poly_start)
        self.assertRtolEqual(poly_end_expected, poly_end)
        self.assertRtolEqual(sl_expected, sl)

    def test_cartesian_to_frenet_should_return_right_value_when_shape_has_more_than_three_dimensions(self):
        n = 2
        bs = 5000
        k1 = 20
        k2 = 40
        dim = 2

        pt, poly_line = gen_data_multi_d(n, bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        poly_start_expected, poly_end_expected, sl_expected = self.golden_op(pt, poly_line)
        poly_start, poly_end, sl = mx_driving.cartesian_to_frenet(pt, poly_line)

        poly_start_expected = poly_start_expected.cpu().numpy()
        poly_end_expected = poly_end_expected.cpu().numpy()
        sl_expected = sl_expected.cpu().numpy()

        poly_start = poly_start.cpu().numpy()
        poly_end = poly_end.cpu().numpy()
        sl = sl.cpu().numpy()

        self.assertRtolEqual(poly_start_expected, poly_start)
        self.assertRtolEqual(poly_end_expected, poly_end)
        self.assertRtolEqual(sl_expected, sl)

    def test_cartesian_to_frenet_should_raise_error_value_when_k2_is_1(self):
        bs = 5000
        k1 = 20
        k2 = 1
        dim = 2

        pt, poly_line = gen_data(bs, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        try:
            poly_start, poly_end, sl = mx_driving.cartesian_to_frenet(pt, poly_line)
        except Exception as e:
            assert "Number of points in polyline must be greater than 1." in str(e)

    def test_cartesian_to_frenet_should_raise_error_value_when_bs_differs(self):
        bs = 5000
        k1 = 20
        k2 = 1
        dim = 2

        pt, _ = gen_data(bs, k1, k2, dim)
        _, poly_line = gen_data(bs - 1000, k1, k2, dim)
        pt = torch.from_numpy(pt).npu()
        poly_line = torch.from_numpy(poly_line).npu()
        try:
            poly_start, poly_end, sl = mx_driving.cartesian_to_frenet(pt, poly_line)
        except Exception as e:
            assert "Point and poly_line must share the same batch size." in str(e)

if __name__ == "__main__":
    run_tests()
