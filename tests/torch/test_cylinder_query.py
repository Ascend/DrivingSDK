import unittest


import torch
import torch_npu
import numpy as np
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests
from mx_driving.ops.cylinder_query import cylinder_query


# 'pylint: disable=too-many-arguments,huawei-too-many-arguments
@golden_data_cache(__file__)
def gen_data(B, N, M):
    new_xyz = np.random.randn(B, M, 3).astype(np.float32)
    xyz = np.random.randn(B, N, 3).astype(np.float32)
    rot = np.random.randn(B, M, 9).astype(np.float32)
    return new_xyz, xyz, rot


class TestCylinderQuert(TestCase):
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def cpu_forward_op(self,
        radius,
        hmin,
        hmax,
        nsample,
        new_xyz,
        xyz,
        rot):
        B = xyz.shape[0]
        N = xyz.shape[1]
        M = new_xyz.shape[1]
        
        xyzTrans = xyz[:, None, :, :] - new_xyz[:, :, None, :] # (b, m, n, 3)
        rot = rot.reshape(B, M, 3, 3)
        xyzTrans = (rot[:, :, None, :, :] * xyzTrans[:, :, :, :, None]).sum(-2)
        d2 = xyzTrans[:, :, :, 1] * xyzTrans[:, :, :, 1] + xyzTrans[:, :, :, 2] * xyzTrans[:, :, :, 2]
        h = xyzTrans[:, :, :, 0]
        radius2 = radius ** 2
        
        not_in_cylinder = np.logical_or(np.logical_or(d2 >= radius2, h <= hmin), h >= hmax)
        
        group_idx = np.arange(N, dtype=np.int32).reshape(1, 1, N)
        group_idx = np.tile(group_idx, (B, M, 1))
        group_idx[not_in_cylinder] = N
        
        group_idx = np.sort(group_idx, axis=-1)[..., :nsample]
        group_first = group_idx[..., 0, np.newaxis]  # 对应view(B, M, 1)
        group_first = np.tile(group_first, (1, 1, nsample))
        group_first[group_first == N] = 0
        mask = (group_idx == N)
        group_idx[mask] = group_first[mask]
        return group_idx

    def test_cylinder_query_return_right_value_when_shape_is_all_one(self):
        B = 1
        N = 1
        M = 1
        radius = 10
        hmin = -100
        hmax = 100
        nsample = 1
        new_xyz, xyz, rot = gen_data(B, N, M)
        expected_output = self.cpu_forward_op(radius, hmin, hmax, nsample, new_xyz, xyz, rot)

        output = cylinder_query(radius,
                                hmin,
                                hmax,
                                nsample,
                                torch.from_numpy(new_xyz).npu(),
                                torch.from_numpy(xyz).npu(),
                                torch.from_numpy(rot).npu())
        output = output.cpu().numpy()
        self.assertRtolEqual(expected_output, output)
    
    def test_cylinder_query_should_return_right_value_when_shape_is_align_to_8(self):
        B = 8
        N = 128
        M = 32
        radius = 10
        hmin = -100
        hmax = 100
        nsample = 8
        new_xyz, xyz, rot = gen_data(B, N, M)
        expected_output = self.cpu_forward_op(radius, hmin, hmax, nsample, new_xyz, xyz, rot)
        output = cylinder_query(radius,
                                hmin,
                                hmax,
                                nsample,
                                torch.from_numpy(new_xyz).npu(),
                                torch.from_numpy(xyz).npu(),
                                torch.from_numpy(rot).npu())
        output = output.cpu().numpy()
        self.assertRtolEqual(expected_output, output)

    def test_cylinder_query_should_return_right_value_when_shape_is_not_align(self):
        B = 7
        N = 129
        M = 31
        radius = 10
        hmin = -100
        hmax = 100
        nsample = 9
        new_xyz, xyz, rot = gen_data(B, N, M)
        expected_output = self.cpu_forward_op(radius, hmin, hmax, nsample, new_xyz, xyz, rot)
        output = cylinder_query(radius,
                                hmin,
                                hmax,
                                nsample,
                                torch.from_numpy(new_xyz).npu(),
                                torch.from_numpy(xyz).npu(),
                                torch.from_numpy(rot).npu())
        output = output.cpu().numpy()
        self.assertRtolEqual(expected_output, output)

    def test_cylinder_query_should_return_right_value_when_N_is_1000(self):
        B = 7
        N = 1000
        M = 31
        radius = 10
        hmin = -100
        hmax = 100
        nsample = 9
        new_xyz, xyz, rot = gen_data(B, N, M)

        expected_output = self.cpu_forward_op(radius, hmin, hmax, nsample, new_xyz, xyz, rot)
        output = cylinder_query(radius,
                                hmin,
                                hmax,
                                nsample,
                                torch.from_numpy(new_xyz).npu(),
                                torch.from_numpy(xyz).npu(),
                                torch.from_numpy(rot).npu())
        output = output.cpu().numpy()
        self.assertRtolEqual(expected_output, output)
        
    def test_cylinder_query_should_return_right_value_when_N_is_20000_and_M_is_1024_and_nsample_is_64(self):
        B = 7
        N = 20000
        M = 1024
        radius = 10
        hmin = -0.5
        hmax = 0.5
        nsample = 64
        new_xyz, xyz, rot = gen_data(B, N, M)

        expected_output = self.cpu_forward_op(radius, hmin, hmax, nsample, new_xyz, xyz, rot)
        output = cylinder_query(radius,
                                hmin,
                                hmax,
                                nsample,
                                torch.from_numpy(new_xyz).npu(),
                                torch.from_numpy(xyz).npu(),
                                torch.from_numpy(rot).npu())
        output = output.cpu().numpy()
        self.assertRtolEqual(expected_output, output)

    def test_cylinder_query_should_raise_error_value_when_nsample_is_larger_than_N(self):
        B = 7
        N = 129
        M = 31
        radius = 10
        hmin = -100
        hmax = 100
        nsample = 110
        new_xyz, xyz, rot = gen_data(B, N, M)

        try:
            output = cylinder_query(radius,
                                hmin,
                                hmax,
                                nsample,
                                torch.from_numpy(new_xyz).npu(),
                                torch.from_numpy(xyz).npu(),
                                torch.from_numpy(rot).npu())
        except Exception as e:
            assert "The value of nsample should be greater than the number of points in the tensor xyz." in str(e)

    def test_cylinder_query_should_raise_error_value_when_hmin_is_equal_to_hmax(self):
        B = 7
        N = 129
        M = 31
        radius = 10
        hmin = 11
        hmax = 11
        nsample = 110
        new_xyz, xyz, rot = gen_data(B, N, M)

        try:
            output = cylinder_query(radius,
                                hmin,
                                hmax,
                                nsample,
                                torch.from_numpy(new_xyz).npu(),
                                torch.from_numpy(xyz).npu(),
                                torch.from_numpy(rot).npu())
        except Exception as e:
            assert "The value of hmin needs to be less than the value of hmax." in str(e)

    def test_cylinder_query_should_raise_error_value_when_hmin_is_larger_than_hmax(self):
        B = 7
        N = 129
        M = 31
        radius = 10
        hmin = 11
        hmax = 10
        nsample = 110
        new_xyz, xyz, rot = gen_data(B, N, M)

        try:
            output = cylinder_query(radius,
                                hmin,
                                hmax,
                                nsample,
                                torch.from_numpy(new_xyz).npu(),
                                torch.from_numpy(xyz).npu(),
                                torch.from_numpy(rot).npu())
        except Exception as e:
            assert "The value of hmin needs to be less than the value of hmax." in str(e)
    
    def test_cylinder_query_should_raise_error_value_when_nsample_is_zero(self):
        B = 7
        N = 129
        M = 31
        radius = 10
        hmin = -100
        hmax = 100
        nsample = 0
        new_xyz, xyz, rot = gen_data(B, N, M)

        try:
            output = cylinder_query(radius,
                                hmin,
                                hmax,
                                nsample,
                                torch.from_numpy(new_xyz).npu(),
                                torch.from_numpy(xyz).npu(),
                                torch.from_numpy(rot).npu())
        except Exception as e:
            assert "The value of nsample should be greater than 0." in str(e)
            
    def test_cylinder_query_should_raise_error_value_when_nsample_is_less_than_zero(self):
        B = 7
        N = 129
        M = 31
        radius = 10
        hmin = -100
        hmax = 100
        nsample = -1
        new_xyz, xyz, rot = gen_data(B, N, M)

        try:
            output = cylinder_query(radius,
                                hmin,
                                hmax,
                                nsample,
                                torch.from_numpy(new_xyz).npu(),
                                torch.from_numpy(xyz).npu(),
                                torch.from_numpy(rot).npu())
        except Exception as e:
            assert "The value of nsample should be greater than 0." in str(e)

if __name__ == "__main__":
    run_tests()