import unittest

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving.point
from mx_driving import Voxelization


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestHardVoxelize(TestCase):
    seed = 1024
    point_nums = [1, 7, 6134, 99999]
    np.random.seed(seed)

    @golden_data_cache(__file__)
    def gen(self, point_num):
        x = 108 * np.random.rand(point_num) - 54 
        y = 108 * np.random.rand(point_num) - 54 
        z = 10 * np.random.rand(point_num) - 5
        return np.stack([x, y, z], axis=-1)

    def npu_hard_voxelize(self, points):
        points_npu = torch.from_numpy(points.astype(np.float32)).npu()
        vlz1 = Voxelization(
            [0.075, 0.075, 0.2], [-54, -54, -5, 54, 54, 5], 10, 1000
        )
        cnt1, pts1, voxs1, num_per_vox1 = vlz1(points_npu)
        vlz = mx_driving.point.Voxelization(
            [0.075, 0.075, 0.2], [-54, -54, -5, 54, 54, 5], 10, 1000
        )
        cnt, pts, voxs, num_per_vox = vlz(points_npu)
        return cnt, voxs.cpu().numpy(), cnt1, voxs1.cpu().numpy()

    @golden_data_cache(__file__)
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


    def test_hard_voxelize(self):
        for point_num in self.point_nums:
            voxels = self.gen(point_num)
            cnt_cpu, res_cpu = self.golden_hard_voxelize(voxels)
            cnt_npu, res_npu, cnt_npu1, res_npu1 = self.npu_hard_voxelize(voxels)
            self.assertRtolEqual(cnt_cpu, cnt_npu)
            self.assertRtolEqual(cnt_cpu, cnt_npu1)
            self.assertRtolEqual(res_cpu, res_npu)
            self.assertRtolEqual(res_cpu, res_npu1)


if __name__ == "__main__":
    run_tests()
