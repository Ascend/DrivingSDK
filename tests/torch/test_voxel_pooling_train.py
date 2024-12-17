import copy
import unittest

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving.point
from mx_driving import npu_voxel_pooling_train


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


# pylint: disable=too-many-arguments,huawei-too-many-arguments
@golden_data_cache(__file__)
def voxel_pooling_train_cpu_forward(
    batch_size, num_points, num_channels, num_voxel_x, num_voxel_y, num_voxel_z, geom_xyz, input_features
):
    dtype = input_features.dtype
    pos_memo = torch.zeros((batch_size, num_points, 3), dtype=torch.int32) * -1
    output_features = torch.zeros((batch_size, num_voxel_y, num_voxel_x, num_channels), dtype=dtype)
    for i in range(batch_size):
        for j in range(num_points):

            sample_x = geom_xyz[i][j][0]
            sample_y = geom_xyz[i][j][1]
            sample_z = geom_xyz[i][j][2]

            if sample_x < 0 or sample_x >= num_voxel_x:
                continue
            if sample_y < 0 or sample_y >= num_voxel_y:
                continue
            if sample_z < 0 or sample_z >= num_voxel_z:
                continue

            pos_memo[i][j][0] = i
            pos_memo[i][j][1] = geom_xyz[i][j][1]
            pos_memo[i][j][2] = geom_xyz[i][j][0]

            for k in range(num_channels):
                output_features[i][sample_y][sample_x][k] += input_features[i][j][k]
    return pos_memo, output_features.permute(0, 3, 1, 2)


@golden_data_cache(__file__)
def voxel_pooling_train_cpu_backward(pos, result_cpu, grad_features):
    features_shape = grad_features.shape
    mask = (pos != -1)[..., 0]

    grad_features = grad_features.reshape(grad_features.shape[0], -1, grad_features.shape[-1])

    grad_features[mask] = result_cpu[pos[mask][..., 0].long(), :, pos[mask][..., 1].long(), pos[mask][..., 2].long()]

    grad_features = grad_features.reshape(features_shape)
    return grad_features


class TestVoxelPoolingTrain(TestCase):
    def cpu_to_exec(self, geom_xyz, input_features, voxel_num):
        batch_size = input_features.shape[0]
        num_points = input_features.shape[1]
        num_channels = input_features.shape[2]
        pos, result = voxel_pooling_train_cpu_forward(
            batch_size, num_points, num_channels, voxel_num[0], voxel_num[1], voxel_num[2], geom_xyz, input_features
        )

        pos_memo = pos
        grad_features_cpu = torch.zeros_like(input_features)
        grad_features_cpu = voxel_pooling_train_cpu_backward(pos_memo, result, grad_features_cpu)

        return pos, result, grad_features_cpu

    def npu_to_exec(self, geom_xyz, input_features, voxel_num):
        result1 = npu_voxel_pooling_train(geom_xyz, input_features, voxel_num)
        input_features2 = input_features.clone().detach().requires_grad_()
        result2 = mx_driving.point.npu_voxel_pooling_train(geom_xyz, input_features2, voxel_num)

        result1.backward(result1)
        result2.backward(result2)
        grad_features_npu1 = input_features.grad
        grad_features_npu2 = input_features2.grad

        return result1, result2, grad_features_npu1, grad_features_npu2

    @golden_data_cache(__file__)
    def gen_data(self, geom_shape, feature_shape, coeff, batch_size, num_channels, dtype):
        geom_xyz = torch.rand(geom_shape) * coeff
        geom_xyz = geom_xyz.reshape(batch_size, -1, 3)
        geom_xyz[:, :, 2] /= 100
        geom_xyz_cpu = geom_xyz.int()
        geom_xyz_npu = geom_xyz_cpu.npu()
        features = torch.rand(feature_shape, dtype=dtype) - 0.5
        features_cpu = features.reshape(batch_size, -1, num_channels)
        features_npu = features_cpu.npu()
        features_npu.requires_grad = True
        return geom_xyz_cpu, features_cpu, geom_xyz_npu, features_npu

    @unittest.skipIf(DEVICE_NAME != "Ascend910B", "OP `VoxelPoolingTrain` is only supported on 910B, skip this ut!")
    def test_voxel_pooling_train(self):
        torch.npu.set_device("npu:0")
        types = [
            torch.float32,
        ]
        batch_size_list = [1, 2]
        num_channels_list = [32, 80]
        shape_list = [[30, 25], [25, 12, 40], [20]]
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
                            geom_shape, feature_shape, coeff, batch_size, num_channel, dtype
                        )
                        pos, cpu_result, cpu_grad_features = self.cpu_to_exec(geom_cpu, feature_cpu, voxel_num)
                        npu_result1, npu_result2, npu_grad_features1, npu_grad_features2 = self.npu_to_exec(geom_npu, feature_npu, voxel_num)

                        cpu_result = cpu_result.numpy()
                        npu_result1 = npu_result1.detach().cpu().numpy()
                        npu_result2 = npu_result2.detach().cpu().numpy()
                        
                        self.assertRtolEqual(cpu_result, npu_result1)
                        self.assertRtolEqual(cpu_result, npu_result2)
                        
                        cpu_grad_features = cpu_grad_features.numpy()
                        npu_grad_features1 = npu_grad_features1.cpu().numpy()
                        npu_grad_features2 = npu_grad_features2.cpu().numpy()
                        
                        self.assertRtolEqual(cpu_grad_features, npu_grad_features1)
                        self.assertRtolEqual(cpu_grad_features, npu_grad_features2)
                        

if __name__ == "__main__":
    run_tests()
