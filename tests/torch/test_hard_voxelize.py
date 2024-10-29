import unittest

import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import mx_driving._C

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestHardVoxelize(TestCase):
    seed = 1024
    point_nums = [1, 7, 6134, 99999]
    np.random.seed(seed)

    def gen(self, point_num):
        x = 108 * np.random.rand(point_num) - 54 
        y = 108 * np.random.rand(point_num) - 54 
        z = 10 * np.random.rand(point_num) - 5
        return np.stack([x, y, z], axis=-1)

    def npu_hard_voxelize(self, points):
        points_npu = torch.from_numpy(points.astype(np.float32)).npu()
        cnt, pts, voxs, num_per_vox = mx_driving._C.hard_voxelize(
            points_npu, [0.075, 0.075, 0.2], [-54, -54, -5, 54, 54, 5], 10, 1000
        )
        return cnt, voxs.cpu().numpy()

    def golden_hard_voxelize(self, points):
        point_num = points.shape[0]
        gridx = 1440
        gridy = 1440
        gridz = 50
        points = points.astype(np.float64)
        coorx = np.floor((points[:, 0] + 54) / 0.075).astype(np.int32)
        coory = np.floor((points[:, 1] + 54) / 0.075).astype(np.int32)
        coorz = np.floor((points[:, 2] + 5) / 0.2).astype(np.int32)
        result = []
        seen = set()
        for i in range(point_num):
            x, y, z = coorx[i], coory[i], coorz[i]
            if x >= 0 and x < gridx and y >= 0 and y < gridy and z >= 0 and z < gridz:
                code = (x << 19) | (y << 8) | z
                if code not in seen:
                    seen.add(code)
                    result.append([x, y, z])
                if len(seen) == 1000:
                    break
        
        return len(result), np.array(result)



    @unittest.skipIf(DEVICE_NAME != "Ascend910B", "OP `PointToVoxel` is only supported on 910B, skip this ut!")
    def test_hard_voxelize(self):
        for point_num in self.point_nums:
            voxels = self.gen(point_num)
            cnt_cpu, res_cpu = self.golden_hard_voxelize(voxels)
            cnt_npu, res_npu = self.npu_hard_voxelize(voxels)
            self.assertRtolEqual(cnt_cpu, cnt_npu)
            self.assertRtolEqual(res_cpu, res_npu)


if __name__ == "__main__":
    run_tests()
