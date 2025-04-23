# Copyright (c) 2024, Huawei Technologies.All rights reserved.
# Copyright 2021 Yan Yan
"""Compare results between different algos:
CPU: simple gather-mm-scatter
Native: Fused gather-mm-scatter
ImplicitGemm: implicit gemm
"""

import time
from pathlib import Path
import numpy as np
import torch
import torch_npu
from torch import nn
from torch_npu.testing.testcase import TestCase, run_tests
from data_cache import golden_data_cache
from mx_driving.spconv import SparseSequential, SparseConvTensor, SubMConv3d


@golden_data_cache(__file__)
def generate_sparse_data(num_points, spatial_shape, in_channels):
    bs = len(num_points)
    total_points = sum(num_points)
    features = np.random.uniform(0, 5, (total_points, in_channels))
    indices = []
    batch_idx = 0
    for num_point in num_points:
        batch_indices = []
        batch_indices.append(np.ones((2 * num_point, 1)) * batch_idx)
        for spatial_size in spatial_shape:
            idx = np.random.uniform(0, spatial_size, (2 * num_point, 1)).astype(np.int32)
            batch_indices.append(idx)
        
        batch_indices = np.concatenate(batch_indices, axis=1)
        idx_unique = np.unique(batch_indices, axis=0)
        indices.append(idx_unique[:num_point])
        batch_idx += 1
        
    indices = np.concatenate(indices, axis=0)
    return torch.from_numpy(features).float(), torch.from_numpy(indices).int()


def generate_map(coors, spatial_shape, bs):
    spatial_shape1 = (spatial_shape[1] * spatial_shape[0])
    new_coors1 = spatial_shape1 * coors[:, 0] + spatial_shape[1] * coors[:, 1] + coors[:, 2]
    map1 = torch.full((spatial_shape1 * bs, ), -1, dtype=torch.int32, device=coors.device)
    map1[new_coors1] = torch.arange(new_coors1.numel(), dtype=torch.int32, device=coors.device)
    mask = map1 != -1
    map1_unqiue_size = mask.sum()
    map1[mask] = torch.arange(map1_unqiue_size, dtype=torch.int32, device=coors.device)

    map2 = torch.full((map1_unqiue_size, spatial_shape[2]), -1, dtype=torch.int32, device=coors.device)
    map2[map1[new_coors1], coors[:, 3]] = torch.arange(new_coors1.numel(), dtype=torch.int32, device=coors.device)
    return map1, map2


# pylint: disable=too-many-arguments,huawei-too-many-arguments
@golden_data_cache(__file__)
def get_golden_output(features, indices, weights, bias, batch_size, in_channels,
                      out_channels, kernel_size, out_spatial_shape):
    map1, map2 = generate_map(indices, out_spatial_shape, batch_size)
    M = torch.zeros((features.shape[0], kernel_size, kernel_size, kernel_size, in_channels), device=features.device)
    weight_flatten = weights.reshape((kernel_size * kernel_size * kernel_size * in_channels, out_channels))

    min_x_idx = indices[:, 1] - kernel_size // 2
    min_y_idx = indices[:, 2] - kernel_size // 2
    min_z_idx = indices[:, 3] - kernel_size // 2

    kernel_offset = torch.arange(kernel_size, device=features.device)
    k0 = torch.broadcast_to(kernel_offset.reshape((kernel_size, 1, 1)), (kernel_size, kernel_size, kernel_size))
    k1 = torch.broadcast_to(kernel_offset.reshape((1, kernel_size, 1)), (kernel_size, kernel_size, kernel_size))
    k2 = torch.broadcast_to(kernel_offset.reshape((1, 1, kernel_size)), (kernel_size, kernel_size, kernel_size))
    
    x_idx = min_x_idx[:, None, None, None] + k0[None, :]
    y_idx = min_y_idx[:, None, None, None] + k1[None, :]
    z_idx = min_z_idx[:, None, None, None] + k2[None, :]

    mask = (x_idx >= 0) * (y_idx >= 0) * (z_idx >= 0) * (x_idx < out_spatial_shape[0]) * (y_idx < out_spatial_shape[1]) * (z_idx < out_spatial_shape[2])
    
    map1_idx = (indices[:, 0, None, None, None] * out_spatial_shape[1] * out_spatial_shape[0] + x_idx * out_spatial_shape[1] + y_idx)[mask]
    map2_idx = z_idx[mask]
    
    map1_val = map1[map1_idx]
    mask1 = map1_val != -1
    map1_val = map1_val[mask1]
    map2_idx = map2_idx[mask1]
    mask[mask.clone()] = mask1
    
    points_offset = map2[map1_val, map2_idx]
    mask2 = points_offset != -1
    mask[mask.clone()] = mask2
    
    M[mask] = features[points_offset[mask2], :]
    out = M.reshape(features.shape[0], -1) @ weight_flatten + bias.reshape(1, -1)
    return out


# pylint: disable=too-many-arguments,huawei-too-many-arguments
def get_output(num_points, batch_size, in_channels, out_channels,
        kernel_size, spatial_shape):
    features, indices = generate_sparse_data(num_points, spatial_shape, in_channels)
    features, indices = features.npu(), indices.npu()
    net = SubMConv3d(in_channels, out_channels, kernel_size).npu()
    x = SparseConvTensor(features, indices, spatial_shape, batch_size)
    golden_output = get_golden_output(features, indices, net.weight.data, net.bias.data, batch_size,
        in_channels, out_channels, kernel_size, spatial_shape)
    res = net(x).features
    return res.detach().cpu().numpy(), golden_output.detach().cpu().numpy()


class TestSubmSparseConv3d(TestCase):
    def test_model_case1(self):
        num_points = [61557]
        out_spatial_shape = [1440, 1440, 41]
        in_channels = 16
        out_channels = 32
        kernel_size = 3
        batch_size = len(num_points)

        res, golden = get_output(num_points, batch_size, in_channels, out_channels, kernel_size, out_spatial_shape)
        self.assertRtolEqual(golden, res)

    def test_model_case2(self):
        num_points = [38153]
        out_spatial_shape = [1180, 180, 5]
        in_channels = 128
        out_channels = 256
        kernel_size = 3
        batch_size = len(num_points)

        res, golden = get_output(num_points, batch_size, in_channels, out_channels, kernel_size, out_spatial_shape)
        self.assertRtolEqual(golden, res)

    def test_5x5_kernel_case1(self):
        num_points = [38153]
        out_spatial_shape = [1180, 180, 5]
        in_channels = 128
        out_channels = 256
        kernel_size = 5
        batch_size = len(num_points)

        res, golden = get_output(num_points, batch_size, in_channels, out_channels, kernel_size, out_spatial_shape)
        self.assertRtolEqual(golden, res)
    
    def test_5x5_kernel_case2(self):
        num_points = [38153]
        out_spatial_shape = [1180, 180, 5]
        in_channels = 128
        out_channels = 256
        kernel_size = 5
        batch_size = len(num_points)

        res, golden = get_output(num_points, batch_size, in_channels, out_channels, kernel_size, out_spatial_shape)
        self.assertRtolEqual(golden, res)

    def test_large_spatial_shape(self):
        num_points = [23787]
        out_spatial_shape = [3571, 4251, 1062]
        in_channels = 4
        out_channels = 32
        kernel_size = 5
        batch_size = len(num_points)

        res, golden = get_output(num_points, batch_size, in_channels, out_channels, kernel_size, out_spatial_shape)
        self.assertRtolEqual(golden, res)

    def test_unaligned_channel(self):
        num_points = [10000]
        out_spatial_shape = [1180, 180, 5]
        in_channels = 55
        out_channels = 77
        kernel_size = 5
        batch_size = len(num_points)

        res, golden = get_output(num_points, batch_size, in_channels, out_channels, kernel_size, out_spatial_shape)
        self.assertRtolEqual(golden, res)

if __name__ == "__main__":
    np.random.seed(100)
    run_tests()