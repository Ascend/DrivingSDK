import math
import random
import unittest

import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving.point
from mx_driving import Voxelization


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


@golden_data_cache(__file__)
def dyn_voxelization_cpu(points, voxel_size, coors_range):
    num_points = points.size(0)

    float_espolin = 1e-9
    if ((voxel_size[0] < float_espolin) or
        (voxel_size[1] < float_espolin) or
        (voxel_size[2] < float_espolin)):
        print("ERROR: voxel size should larger than zero")

    grid_x = round((coors_range[3] - coors_range[0]) / voxel_size[0])
    grid_y = round((coors_range[4] - coors_range[1]) / voxel_size[1])
    grid_z = round((coors_range[5] - coors_range[2]) / voxel_size[2])
    coors = torch.zeros((num_points, 3), dtype=torch.int32)

    for i in range(num_points):
        pts_x = points[i][0]
        pts_y = points[i][1]
        pts_z = points[i][2]
        c_x = math.floor((pts_x - coors_range[0]) / voxel_size[0])
        c_y = math.floor((pts_y - coors_range[1]) / voxel_size[1])
        c_z = math.floor((pts_z - coors_range[2]) / voxel_size[2])
        invalid_cx = c_x < 0 or c_x >= grid_x
        invalid_cy = c_y < 0 or c_y >= grid_y
        invalid_cz = c_z < 0 or c_z >= grid_z
        if (invalid_cx or invalid_cy or invalid_cz):
            coors[i][0] = -1
            coors[i][1] = -1
            coors[i][2] = -1
        else:
            coors[i][0] = c_z
            coors[i][1] = c_y
            coors[i][2] = c_x
    return coors


class TestDynVoxelization(TestCase):
    torch.manual_seed(2024)

    def cpu_to_exec(self, points, coors_range, voxel_size):
        coors = dyn_voxelization_cpu(points, voxel_size, coors_range)
        return coors
    
    def npu_to_exec(self, points, coors_range, voxel_size):
        max_num_points = -1
        dynamic_voxelization_npu = mx_driving.point.Voxelization(voxel_size, coors_range, max_num_points)
        dynamic_voxelization_npu1 = Voxelization(voxel_size, coors_range, max_num_points)
        coors = dynamic_voxelization_npu(points)
        coors1 = dynamic_voxelization_npu1(points)
        return coors, coors1

    @golden_data_cache(__file__)
    def gen_data(self, shape, dtype):
        points_cpu = torch.randint(-20, 100, shape, dtype=dtype)
        points_cpu = points_cpu + torch.rand(shape, dtype=dtype)
        points_npu = points_cpu.npu()
        return points_cpu, points_npu

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `DynVoxelization` is only supported on 910B, skip this ut!")
    def test_dyn_voxelization_general(self):
        dtype = torch.float32
        points_shape_list = [
            [16, 3],
            [351, 3],
            [4021, 4],
            [13542, 4],
            [103584, 5]
        ]
        points_cloud_range_list = [
            [0, -40, -3, 70.4, 40, 1],
            [-10, 12, 0.5, 0.8, 22.3, 20.3],
            [-40.2, 0.8, -10.1, 20.8, 152.3, 65.2],
        ]
        voxel_size = [0.5, 0.5, 0.5]
        for shape in points_shape_list:
            for coors_range in points_cloud_range_list:
                points_cpu, points_npu = self.gen_data(shape, dtype)
                coors_cpu = self.cpu_to_exec(points_cpu, coors_range, voxel_size)
                coors_npu, coors_npu1 = self.npu_to_exec(points_npu, coors_range, voxel_size)
                self.assertRtolEqual(coors_cpu, coors_npu)
                self.assertRtolEqual(coors_cpu, coors_npu1)
                
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `DynVoxelization` is only supported on 910B, skip this ut!")
    def test_dyn_voxelization_boundary(self):
        points_cpu = torch.tensor([[0.9890, 4.6407, -4.9517, -1.3076, 0.2576, -1.4615, -1.6132, -1.8242, 0.4206]])
        voxel_size = [35.47308521738105, 15.469724056371934, 92.16283622466237]
        coors_range = [-266.61751401434685, -490.3904950050411, -672.6572843925751,
                       191.79477406132705, 677.45900318772, 248.5950831410758]
        points_npu = points_cpu.npu()
        coors_cpu = self.cpu_to_exec(points_cpu, coors_range, voxel_size)
        coors_npu, coors_npu1 = self.npu_to_exec(points_npu, coors_range, voxel_size)
        self.assertRtolEqual(coors_cpu, coors_npu)
        self.assertRtolEqual(coors_cpu, coors_npu1)

if __name__ == '__main__':
    run_tests()