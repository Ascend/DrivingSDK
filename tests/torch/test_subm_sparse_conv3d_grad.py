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
from test_subm_sparse_conv3d import get_golden_output as forward_func
from mx_driving.spconv import SparseSequential, SparseConvTensor, SubMConv3d


@golden_data_cache(__file__)
def generate_sparse_data(num_points, spatial_shape, in_channels, out_channels):
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
    grad_out = np.random.uniform(-5, 5, (num_point, out_channels))
    return torch.from_numpy(features).float(), torch.from_numpy(indices).int(), torch.from_numpy(grad_out).float()


# pylint: disable=too-many-arguments,huawei-too-many-arguments
@golden_data_cache(__file__)
def get_golden_output(features, indices, weight, bias, batch_size, in_channels,
                      out_channels, kernel_size, out_spatial_shape, grad_out):
    _, indices_offset, img2col_mat = forward_func(features, indices, weight, bias, batch_size, in_channels,
        out_channels, kernel_size, out_spatial_shape)
    features_grad, weight_grad = subm_sparse_conv3d_grad_cpu(grad_out, img2col_mat, indices_offset, weight, kernel_size, in_channels)
    return features_grad, weight_grad


# pylint: disable=too-many-arguments,huawei-too-many-arguments
@golden_data_cache(__file__)
def subm_sparse_conv3d_grad_cpu(grad_out, img2col_mat, indices_offset, weight, kernel_size, in_channels):
    N, out_channels = grad_out.shape
    weight_grad = img2col_mat.reshape(N, -1).transpose(1, 0) @ grad_out
    img2col_mat_grad = grad_out @ weight.reshape(-1, out_channels).T
    indices_offset = indices_offset.reshape(-1)
    img2col_mat_grad = img2col_mat_grad.reshape(-1, in_channels)

    weight_shape = weight.shape
    weight_grad = weight_grad.view(
        weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3], weight_shape[4]
    )

    mask = (indices_offset != -1)
    valid_indices = torch.nonzero(mask).view(-1)
    indices_offset = torch.index_select(indices_offset, 0, valid_indices)
    img2col_mat_grad = torch.index_select(img2col_mat_grad, 0, valid_indices)
    
    features_grad = torch.zeros(N, in_channels, dtype=grad_out.dtype, device=grad_out.device)
    
    features_grad.scatter_add_(0, indices_offset[:, None].broadcast_to((-1, in_channels)).long(), img2col_mat_grad)
    return features_grad, weight_grad


# pylint: disable=too-many-arguments,huawei-too-many-arguments
def get_output(num_points, batch_size, in_channels, out_channels,
        kernel_size, spatial_shape, dtype):
    features, indices, grad_out = generate_sparse_data(num_points, spatial_shape, in_channels, out_channels)
    features, indices, grad_out = features.to(dtype).npu(), indices.npu(), grad_out.to(dtype).npu()
    net = SubMConv3d(in_channels, out_channels, kernel_size).npu()
    net.weight.data = net.weight.data.to(dtype)
    net.bias.data = net.bias.data.to(dtype) * 0.01
    x = SparseConvTensor(features, indices, spatial_shape, batch_size)
    features.requires_grad = True
    net.weight.requires_grad = True
    
    features_grad, weight_grad = get_golden_output(features, indices, net.weight.data, net.bias.data, batch_size, in_channels,
                      out_channels, kernel_size, spatial_shape, grad_out)
    res = net(x).features
    res.backward(grad_out)
    return features.grad, net.weight.grad, features_grad, weight_grad


class TestSubmSparseConv3d(TestCase):
    def test_3x3x3_fp32(self):
        num_points = [61557]
        out_spatial_shape = [1440, 1440, 41]
        in_channels = 16
        out_channels = 32
        kernel_size = 3
        batch_size = len(num_points)
        dtype = torch.float32

        npu_features_grad, npu_weight_grad, cpu_features_grad, cpu_weight_grad = get_output(num_points, batch_size, in_channels,
            out_channels, kernel_size, out_spatial_shape, dtype)

        self.assertRtolEqual(npu_features_grad, cpu_features_grad)
        self.assertRtolEqual(npu_weight_grad, cpu_weight_grad)

    def test_large_point1(self):
        num_points = [200000]
        out_spatial_shape = [1440, 1440, 41]
        in_channels = 16
        out_channels = 32
        kernel_size = 3
        batch_size = len(num_points)
        dtype = torch.float32

        npu_features_grad, npu_weight_grad, cpu_features_grad, cpu_weight_grad = get_output(num_points, batch_size, in_channels,
            out_channels, kernel_size, out_spatial_shape, dtype)

        self.assertRtolEqual(npu_features_grad, cpu_features_grad)
        self.assertRtolEqual(npu_weight_grad, cpu_weight_grad)
    
    def test_large_point2(self):
        num_points = [200000]
        out_spatial_shape = [1440, 1440, 41]
        in_channels = 16
        out_channels = 32
        kernel_size = 3
        batch_size = len(num_points)
        dtype = torch.float16

        npu_features_grad, npu_weight_grad, cpu_features_grad, cpu_weight_grad = get_output(num_points, batch_size, in_channels,
            out_channels, kernel_size, out_spatial_shape, dtype)

        self.assertRtolEqual(npu_features_grad, cpu_features_grad)
        self.assertRtolEqual(npu_weight_grad, cpu_weight_grad)

    def test_3x3x3_fp16(self):
        num_points = [61557]
        out_spatial_shape = [1440, 1440, 41]
        in_channels = 16
        out_channels = 32
        kernel_size = 3
        batch_size = len(num_points)
        dtype = torch.float16

        npu_features_grad, npu_weight_grad, cpu_features_grad, cpu_weight_grad = get_output(num_points, batch_size, in_channels,
            out_channels, kernel_size, out_spatial_shape, dtype)

        self.assertRtolEqual(npu_features_grad, cpu_features_grad)
        self.assertRtolEqual(npu_weight_grad, cpu_weight_grad)

    def test_model_5x5x5_fp32(self):
        num_points = [61557]
        out_spatial_shape = [1440, 1440, 41]
        in_channels = 16
        out_channels = 32
        kernel_size = 5
        batch_size = len(num_points)
        dtype = torch.float32

        npu_features_grad, npu_weight_grad, cpu_features_grad, cpu_weight_grad = get_output(num_points, batch_size, in_channels,
            out_channels, kernel_size, out_spatial_shape, dtype)

        self.assertRtolEqual(npu_features_grad, cpu_features_grad)
        self.assertRtolEqual(npu_weight_grad, cpu_weight_grad)

    def test_model_5x5x5_fp16(self):
        num_points = [61557]
        out_spatial_shape = [1440, 1440, 41]
        in_channels = 16
        out_channels = 32
        kernel_size = 5
        batch_size = len(num_points)
        dtype = torch.float16

        npu_features_grad, npu_weight_grad, cpu_features_grad, cpu_weight_grad = get_output(num_points, batch_size, in_channels,
            out_channels, kernel_size, out_spatial_shape, dtype)

        self.assertRtolEqual(npu_features_grad, cpu_features_grad)
        self.assertRtolEqual(npu_weight_grad, cpu_weight_grad)

    def test_model_large_channels1(self):
        num_points = [61557]
        out_spatial_shape = [1440, 1440, 41]
        in_channels = 256
        out_channels = 512
        kernel_size = 5
        batch_size = len(num_points)
        dtype = torch.float16

        npu_features_grad, npu_weight_grad, cpu_features_grad, cpu_weight_grad = get_output(num_points, batch_size, in_channels,
            out_channels, kernel_size, out_spatial_shape, dtype)

        self.assertRtolEqual(npu_features_grad, cpu_features_grad)
        self.assertRtolEqual(npu_weight_grad, cpu_weight_grad)

    def test_model_large_channels2(self):
        num_points = [61557]
        out_spatial_shape = [1440, 1440, 41]
        in_channels = 256
        out_channels = 512
        kernel_size = 3
        batch_size = len(num_points)
        dtype = torch.float16

        npu_features_grad, npu_weight_grad, cpu_features_grad, cpu_weight_grad = get_output(num_points, batch_size, in_channels,
            out_channels, kernel_size, out_spatial_shape, dtype)

        self.assertRtolEqual(npu_features_grad, cpu_features_grad)
        self.assertRtolEqual(npu_weight_grad, cpu_weight_grad)

    def test_model_large_channels3(self):
        num_points = [61557]
        out_spatial_shape = [1440, 1440, 41]
        in_channels = 256
        out_channels = 512
        kernel_size = 3
        batch_size = len(num_points)
        dtype = torch.float32

        npu_features_grad, npu_weight_grad, cpu_features_grad, cpu_weight_grad = get_output(num_points, batch_size, in_channels,
            out_channels, kernel_size, out_spatial_shape, dtype)

        self.assertRtolEqual(npu_features_grad, cpu_features_grad)
        self.assertRtolEqual(npu_weight_grad, cpu_weight_grad)
    
    def test_model_large_channels4(self):
        num_points = [61557]
        out_spatial_shape = [1440, 1440, 41]
        in_channels = 256
        out_channels = 512
        kernel_size = 5
        batch_size = len(num_points)
        dtype = torch.float32

        npu_features_grad, npu_weight_grad, cpu_features_grad, cpu_weight_grad = get_output(num_points, batch_size, in_channels,
            out_channels, kernel_size, out_spatial_shape, dtype)

        self.assertRtolEqual(npu_features_grad, cpu_features_grad)
        self.assertRtolEqual(npu_weight_grad, cpu_weight_grad)

if __name__ == "__main__":
    np.random.seed(100)
    run_tests()