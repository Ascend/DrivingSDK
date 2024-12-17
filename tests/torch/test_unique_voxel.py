import unittest

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestUniqueVoxel(TestCase):
    seed = 1024
    np.random.seed(seed)
    point_nums = [1, 7, 6134, 99999]

    @golden_data_cache(__file__)
    def gen(self, point_num):
        x = np.random.randint(0, 1024, (point_num,))
        return x.astype(np.int32)

    @golden_data_cache(__file__)
    def golden_unique(self, voxels):
        res = np.unique(voxels)
        return res.shape[0], np.sort(res)

    def npu_unique(self, voxels):
        voxels_npu = torch.from_numpy(voxels).npu()
        cnt, uni_vox, _, _, _ = mx_driving.unique_voxel(voxels_npu)
        return cnt, uni_vox.cpu().numpy()

    @golden_data_cache(__file__)
    def gen_integration(self, point_num):
        x = np.random.randint(0, 256, (point_num,))
        y = np.random.randint(0, 256, (point_num,))
        z = np.random.randint(0, 256, (point_num,))
        return np.stack([x, y, z], axis=-1).astype(np.int32)

    @golden_data_cache(__file__)
    def golden_integration(self, coords):
        point_num = coords.shape[0]
        res = np.zeros((point_num,), dtype=np.int32)
        for i in range(point_num):
            if coords[i][0] < 0 or coords[i][1] < 0 or coords[i][2] < 0:
                res[i] = -1082130432
            else:
                res[i] = coords[i][0] * 2048 * 256 + coords[i][1] * 256 + coords[i][2]
        uni = np.unique(res)
        uni_sorted = np.sort(uni)
        uni_count = uni.shape[0]

        res = np.zeros((uni_count, 3), dtype=np.int32)
        for i in range(uni_count):
            res[i][0] = uni_sorted[i] // (2048 * 256)
            res[i][1] = (uni_sorted[i] // 256) % 2048
            res[i][2] = uni_sorted[i] % 256

        return uni_count, res

    def npu_integration(self, coords):
        coords_npu = torch.from_numpy(coords.view(np.float32)).npu()
        voxels_npu = mx_driving._C.point_to_voxel(coords_npu, [], [], "XYZ")
        cnt, uni_vox, _, _, _ = mx_driving.unique_voxel(voxels_npu)
        dec = mx_driving._C.voxel_to_point(uni_vox, [], [], "XYZ")
        return cnt, dec.cpu().numpy()

    @unittest.skipIf(DEVICE_NAME != "Ascend910B", "OP `PointToVoxel` is only supported on 910B, skip this ut!")
    def test_unique_voxel(self):
        for point_num in self.point_nums:
            voxels = self.gen(point_num)
            cnt_cpu, res_cpu = self.golden_unique(voxels)
            cnt_npu, res_npu = self.npu_unique(voxels)
            self.assertRtolEqual(cnt_cpu, cnt_npu)
            self.assertRtolEqual(res_cpu, res_npu)

    @unittest.skipIf(DEVICE_NAME != "Ascend910B", "OP `PointToVoxel` is only supported on 910B, skip this ut!")
    def test_unique_voxel_int32(self):
        for point_num in self.point_nums:
            voxels = self.gen(point_num)
            cnt_cpu, res_cpu = self.golden_unique(voxels)
            cnt_npu, res_npu = self.npu_unique(voxels.view(np.float32))
            self.assertRtolEqual(cnt_cpu, cnt_npu)
            self.assertRtolEqual(res_cpu, res_npu)

    @unittest.skipIf(DEVICE_NAME != "Ascend910B", "OP `PointToVoxel` is only supported on 910B, skip this ut!")
    def test_integration(self):
        for point_num in self.point_nums:
            voxels = self.gen_integration(point_num)
            cnt_cpu, res_cpu = self.golden_integration(voxels)
            cnt_npu, res_npu = self.npu_integration(voxels)
            self.assertRtolEqual(cnt_cpu, cnt_npu)
            self.assertRtolEqual(res_cpu, res_npu)


if __name__ == "__main__":
    run_tests()
