import unittest
import copy
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
import ads_c
import numpy as np
import ads.common

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def voxel_pooling_train_cpu(batch_size, num_points, num_channels, num_voxel_x,
                            num_voxel_y, num_voxel_z, geom_xyz, input_features):
    dtype = input_features.dtype
    pos_memo = torch.zeros((batch_size, num_points, 3), dtype=torch.int32) * -1
    output_features = torch.zeros((batch_size, num_voxel_y, num_voxel_x, num_channels), dtype=dtype)
    for i in range(batch_size):
        for j in range(num_points):
            
            pos_memo[i][j][0] = i
            pos_memo[i][j][1] = geom_xyz[i][j][1]
            pos_memo[i][j][2] = geom_xyz[i][j][0]

            sample_x = geom_xyz[i][j][0]
            sample_y = geom_xyz[i][j][1]
            sample_z = geom_xyz[i][j][2]

            if ((sample_x < 0 or sample_x >= num_voxel_x) or
                (sample_y < 0 or sample_y >= num_voxel_y) or
                (sample_z < 0 or sample_z >= num_voxel_z)):
                continue  

            for k in range(num_channels):
                output_features[i][sample_y][sample_x][k] += input_features[i][j][k]
    return pos_memo, output_features.permute(0, 3, 1, 2)


class TestVoxelPoolingTrain(TestCase):
    def cpu_to_exec(self, geom_xyz, input_features, voxel_num):
        batch_size = input_features.shape[0]
        num_points = input_features.shape[1]
        num_channels = input_features.shape[2]
        pos, result = voxel_pooling_train_cpu(batch_size, num_points, num_channels, voxel_num[0],
                                         voxel_num[1], voxel_num[2], geom_xyz, input_features)
        return pos, result

    def npu_to_exec(self, geom_xyz, input_features, voxel_num):
        result = ads.common.npu_voxel_pooling_train(geom_xyz, input_features, voxel_num)
        return result
    
    def gen_data(self, geom_shape, feature_shape, coeff, batch_size, num_channels, dtype):
        geom_xyz = torch.rand(geom_shape) * coeff
        geom_xyz = geom_xyz.reshape(batch_size, -1, 3)
        geom_xyz[:, :, 2] /= 100
        geom_xyz_cpu = geom_xyz.int()
        geom_xyz_npu = geom_xyz_cpu.npu()
        features = torch.rand(feature_shape, dtype=dtype) - 0.5
        features_cpu = features.reshape(batch_size, -1, num_channels)
        features_npu = features_cpu.npu()
        return geom_xyz_cpu, features_cpu, geom_xyz_npu, features_npu
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `DynVoxelization` is only supported on 910B, skip this ut!")
    def test_voxel_pooling_train(self):
        torch.npu.set_device('npu:0')
        types = [torch.float32, ]
        batch_size_list = [1, 2]
        num_channels_list = [32, 80]
        shape_list = [
            [30, 25],
            [25, 12, 40],
            [20]
        ]
        coeff = 90
        voxel_num = [128, 128, 1]
        # test
        for dtype in types:
            for batch_size in batch_size_list:
                for num_channel in num_channels_list:
                    for shape in shape_list:
                        shape.insert(0, batch_size)
                        geom_shape = copy.deepcopy(shape)
                        feature_shape = copy.deepcopy(shape)
                        feature_shape.append(num_channel)
                        geom_shape.append(3)
                        geom_cpu, feature_cpu, geom_npu, feature_npu = self.gen_data(
                            geom_shape, feature_shape, coeff, batch_size, num_channel, dtype)
                        pos, result_cpu = self.cpu_to_exec(geom_cpu, feature_cpu, voxel_num)
                        result_npu = self.npu_to_exec(geom_npu, feature_npu, voxel_num)
                        result_cpu = result_cpu.numpy()
                        result_npu = result_npu.cpu().numpy()
                        print("npu shape ", result_npu.shape)
                        print("cpu shape ", result_cpu.shape)
                        self.assertRtolEqual(result_cpu, result_npu)

if __name__ == '__main__':
    run_tests()