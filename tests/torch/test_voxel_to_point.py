import unittest

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving._C


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestVoxelToPoint(TestCase):
    seed = 1024
    np.random.seed(seed)
    point_nums = [1, 7, 6134, 99999]

    @golden_data_cache(__file__)
    def gen(self, point_num):
        x = np.random.randint(0, 10240, (point_num,))
        return x.astype(np.int32)

    @golden_data_cache(__file__)
    def golden_decode(self, voxels):
        point_num = voxels.shape[0]
        res = np.zeros((point_num, 3), dtype=np.int32)
        for i in range(point_num):
            res[i][0] = voxels[i] // (2048 * 256)
            res[i][1] = (voxels[i] // 256) % 2048
            res[i][2] = voxels[i] % 256
        return res

    def npu_decode(self, voxels):
        voxels_npu = torch.from_numpy(voxels.view(np.int32)).npu()
        return mx_driving._C.voxel_to_point(voxels_npu, [], [], "XYZ").cpu().numpy()

    @unittest.skipIf(DEVICE_NAME != "Ascend910B", "OP `PointToVoxel` is only supported on 910B, skip this ut!")
    def test_point_to_voxel(self):
        for point_num in self.point_nums:
            voxels = self.gen(point_num)
            res_cpu = self.golden_decode(voxels)
            res_npu = self.npu_decode(voxels)
            self.assertRtolEqual(res_cpu, res_npu)


if __name__ == "__main__":
    run_tests()
