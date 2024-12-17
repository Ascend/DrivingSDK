import unittest

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving._C


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestPointToVoxel(TestCase):
    seed = 1024
    np.random.seed(seed)
    point_nums = [1, 7, 6134, 99999]

    @golden_data_cache(__file__)
    def gen(self, point_num):
        x = np.random.randint(-1, 256, (point_num,))
        y = np.random.randint(-1, 256, (point_num,))
        z = np.random.randint(-1, 256, (point_num,))
        return np.stack([x, y, z], axis=-1).astype(np.int32)

    @golden_data_cache(__file__)
    def golden_encode(self, coords):
        point_num = coords.shape[0]
        res = np.zeros((point_num,), dtype=np.int32)
        for i in range(point_num):
            if coords[i][0] < 0 or coords[i][1] < 0 or coords[i][2] < 0:
                res[i] = -1082130432
            else:
                res[i] = coords[i][0] * 2048 * 256 + coords[i][1] * 256 + coords[i][2]
        return res

    def npu_encode(self, coords):
        coords_npu = torch.from_numpy(coords.view(np.float32)).npu()
        return mx_driving._C.point_to_voxel(coords_npu, [], [], "XYZ").cpu().numpy().view(np.int32)

    @unittest.skipIf(DEVICE_NAME != "Ascend910B", "OP `PointToVoxel` is only supported on 910B, skip this ut!")
    def test_point_to_voxel(self):
        for point_num in self.point_nums:
            coords = self.gen(point_num)
            golden_encode = self.golden_encode(coords)
            npu_encode = self.npu_encode(coords)
            self.assertRtolEqual(golden_encode, npu_encode)


if __name__ == "__main__":
    run_tests()
