import unittest

import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import ads_c

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestUniqueVoxel(TestCase):
    seed = 1024
    point_nums = [1, 7, 6134, 99999]

    def gen(self, point_num):
        x = np.random.randint(0, 1024, (point_num,))
        return x.astype(np.int32)

    def golden_unique(self, voxels):
        res = np.unique(voxels)
        return res.shape[0], np.sort(res)

    def npu_unique(self, voxels):
        voxels_npu = torch.from_numpy(voxels.view(np.float32)).npu()
        cnt, uni_vox, _, _ = ads_c.unique_voxel(voxels_npu)
        return cnt, uni_vox.cpu().numpy() 

    @unittest.skipIf(DEVICE_NAME != "Ascend910B", "OP `PointToVoxel` is only supported on 910B, skip this ut!")
    def test_unique_voxel(self):
        for point_num in self.point_nums:
            voxels = self.gen(point_num)
            cnt_cpu, res_cpu = self.golden_unique(voxels)
            cnt_npu, res_npu = self.npu_unique(voxels)
            self.assertRtolEqual(cnt_cpu, cnt_npu)
            self.assertRtolEqual(res_cpu, res_npu)


if __name__ == "__main__":
    run_tests()
