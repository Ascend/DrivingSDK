import unittest

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving._C


np.random.seed(2024)
DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


@golden_data_cache(__file__)
def cpu_gen_inputs(args):
    m, c_out, n, c_in, num_total_grids, num_max_sum_points = args

    np_grad_new_features = np.random.rand(m, c_out).astype(np.float32)
    np_point_cnt_of_grid = np.random.randint(1, 8, (m, num_total_grids)).astype(np.int32)
    np_grouped_idxs = np.column_stack((
        np.random.randint(0, n, num_max_sum_points),
        np.random.randint(0, m, num_max_sum_points),
        np.random.randint(0, num_total_grids, num_max_sum_points)
    )).astype(np.int32)
    np_grad_support_features = np.zeros((n, c_in)).astype(np.float32)

    return np_grad_new_features, np_point_cnt_of_grid, np_grouped_idxs, np_grad_support_features


class TestVecPoolGrad(TestCase):
    @golden_data_cache(__file__)
    def golden_vec_pool_backward(self, grad_new_features, point_cnt_of_grid, grouped_idxs, grad_support_features):
        num_c_out = grad_new_features.shape[1]
        num_total_grids = point_cnt_of_grid.shape[1]
        num_c_each_grid = num_c_out // num_total_grids
        for i in range(grouped_idxs.shape[0]):
            idx_of_support_xyz = grouped_idxs[i, 0]
            idx_of_new_xyz = grouped_idxs[i, 1]
            idx_of_grid_idx = grouped_idxs[i, 2]

            num_total_pts = point_cnt_of_grid[idx_of_new_xyz, idx_of_grid_idx]
            cur_grad = 1 / max(num_total_pts, 1)
            for j in range(grad_support_features.shape[1]):
                chl_idx_of_cin = j % num_c_each_grid
                grad_support_features[idx_of_support_xyz][j] += \
                    grad_new_features[idx_of_new_xyz][idx_of_grid_idx * num_c_each_grid + chl_idx_of_cin] * cur_grad

        return grad_support_features

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `VecPoolBackward` is only supported on 910B, skip this ut!")
    def test_vec_pool_backward(self):
        args_list = [
            [5, 8, 30, 11, 1, 4],
            [10, 18, 25, 21, 2, 9],
            [15, 28, 20, 31, 3, 14],
            [20, 38, 15, 41, 4, 19],
            [25, 48, 10, 51, 5, 24],
            [30, 58, 5, 61, 6, 29]
        ]

        for args in args_list: 
            m, c_out, n, c_in, num_total_grids, num_max_sum_points = args
            np_grad_new_features, np_point_cnt_of_grid, np_grouped_idxs, np_grad_support_features = cpu_gen_inputs(args)

            torch_grad_new_features = torch.from_numpy(np_grad_new_features).npu()
            torch_point_cnt_of_grid = torch.from_numpy(np_point_cnt_of_grid).npu()
            torch_grouped_idxs = torch.from_numpy(np_grouped_idxs).npu()

            golden_grad_support_features = self.golden_vec_pool_backward(
                np_grad_new_features, np_point_cnt_of_grid, np_grouped_idxs, np_grad_support_features)
            real_grad_support_features = mx_driving._C.vec_pool_backward(
                torch_grad_new_features, torch_point_cnt_of_grid, torch_grouped_idxs, n, c_in)

            self.assertRtolEqual(golden_grad_support_features, real_grad_support_features.cpu().numpy())


if __name__ == "__main__":
    run_tests()
