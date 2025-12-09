# Copyright (c) 2024, Huawei Technologies.All rights reserved.

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
from mx_driving.spconv import SparseSequential, SparseConvTensor, SparseConv3d

# pylint: disable=too-many-arguments,huawei-too-many-arguments
@golden_data_cache(__file__)
def sparse_conv3d_gloden(
        features,
        indices,
        weight,
        spatial_shape,
        out_channels,
        batch_size,
        kernel_size,
        stride,
        padding,
        ):
    features = features.cpu()
    indices = indices.cpu()
    weight = weight.cpu()
    k0, k1, k2 = kernel_size
    out_spatial_shape = [(spatial_shape[i] + 2 * padding[i] - kernel_size[i]) // stride[i] + 1 for i in range(3)]
    k_size = kernel_size[0] * kernel_size[1] * kernel_size[2]

    k0_offset = torch.arange(k0, device=features.device)
    k1_offset = torch.arange(k1, device=features.device)
    k2_offset = torch.arange(k2, device=features.device)

    k_offset = torch.cartesian_prod(k0_offset - k0 // 2, k1_offset - k1 // 2, k2_offset - k2 // 2)
    zeros = torch.zeros((k_offset.shape[0],1), device=features.device)
    k_offset = torch.cat([zeros, k_offset], dim=1)
    
    indices_offset = (k_offset + indices[:, None, :]).double()
    indices_offset[..., 1] = (indices_offset[..., 1] + padding[0] - kernel_size[0] // 2) / stride[0]
    indices_offset[..., 2] = (indices_offset[..., 2] + padding[1] - kernel_size[1] // 2) / stride[1]
    indices_offset[..., 3] = (indices_offset[..., 3] + padding[2] - kernel_size[2] // 2) / stride[2]
    
    valid_mask1 = ((indices_offset[..., 1] >= 0) & (indices_offset[..., 1] < out_spatial_shape[0]) & \
                  (indices_offset[..., 2] >= 0) & (indices_offset[..., 2] < out_spatial_shape[1]) & \
                  (indices_offset[..., 3] >= 0) & (indices_offset[..., 3] < out_spatial_shape[2]))
    valid_mask2 = (indices_offset.frac() == 0).all(dim=-1)
    valid_mask = valid_mask1 & valid_mask2

    indices_offset = indices_offset[..., 0] * (out_spatial_shape[0] * out_spatial_shape[1] * out_spatial_shape[2]) + \
        indices_offset[..., 1] * (out_spatial_shape[1] * out_spatial_shape[2]) + \
        indices_offset[..., 2] * out_spatial_shape[2] +  indices_offset[..., 3]
    
    indices_offset[~valid_mask] = -1
    indices_offset = torch.flip(indices_offset, (-1,)).int().flatten()

    to_insert = torch.tensor(-1, device=features.device)
    sorted_idx, sorted_idx_to_former_indices = torch.sort(indices_offset.view(torch.float32))
    new_sorted_idx = torch.cat((to_insert.view(1), sorted_idx.view(torch.int32)), 0)
    new_sorted_idx_2 = torch.cat((sorted_idx.view(torch.int32), to_insert.view(1)), 0)

    sub_result = new_sorted_idx - new_sorted_idx_2
    unique_indices_offset = torch.nonzero(sub_result != 0).flatten()

    out_length = unique_indices_offset.shape[0] - 1
    arange_idx = torch.arange(out_length, device=features.device).repeat_interleave(unique_indices_offset[1:] - unique_indices_offset[:-1]).int()
    k_pos = (sorted_idx_to_former_indices[:unique_indices_offset[-1]] % k_size).int()
    input_idx = (sorted_idx_to_former_indices[:unique_indices_offset[-1]] / k_size).int()

    img2col_mat = torch.zeros((out_length, k_size, features.shape[-1]), device=features.device, dtype=features.dtype)
    img2col_mat[arange_idx, k_pos] = features[input_idx]
    out_features = img2col_mat.reshape(out_length, -1).float() @ weight.reshape(-1, out_channels).float()

    return out_features.to(features.dtype)

@golden_data_cache(__file__)
def generate_sparse_data(num_points, spatial_shape, in_channels):
    bs = len(num_points)
    total_points = sum(num_points)
    features = np.random.uniform(-5, 5, (total_points, in_channels))
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

# pylint: disable=too-many-arguments,huawei-too-many-arguments
def get_output(num_points, batch_size, in_channels, out_channels,
        kernel_size, spatial_shape, dtype, stride, padding):
    features, indices = generate_sparse_data(num_points, spatial_shape, in_channels)
    features, indices = features.to(dtype).npu(), indices.npu()
    net = SparseConv3d(in_channels, out_channels, kernel_size, bias=False, stride=stride, padding=padding).npu()
    
    features = features.to(dtype)
    net.weight.data = net.weight.data.to(dtype)
    
    x = SparseConvTensor(features, indices, spatial_shape, batch_size)
    golden_output = sparse_conv3d_gloden(features, indices, net.weight.data,
        spatial_shape, out_channels, batch_size, kernel_size,
        stride, padding)
    res = net(x).features
    return res.detach().cpu().numpy(), golden_output.detach().cpu().numpy()


class TestSparseConv3d(TestCase):
    def do_custom_test(self, num_points, out_spatial_shape, in_channels, out_channels, kernel_size, batch_size, stride, padding):
        res, golden = get_output(num_points, batch_size, in_channels, out_channels,
            kernel_size, out_spatial_shape, torch.float32, stride, padding)
        self.assertRtolEqual(golden, res)
        res, golden = get_output(num_points, batch_size, in_channels, out_channels,
            kernel_size, out_spatial_shape, torch.float16, stride, padding)
        self.assertRtolEqual(golden, res, 1e-2, 1e-2)

    def test(self):
        self.do_custom_test([61557], [1440, 1440, 41], 16, 32, [3,3,3], 1, [2,2,2], [1,1,1])    # bevfusion case
        self.do_custom_test([61557], [1440, 1440, 41], 16, 32, [3,3,3], 1, [1,1,2], [0,0,1])    # bevfusion case
        self.do_custom_test([38153], [1180, 180, 5], 128, 256, [3,3,3], 1, [2,2,2], [1,1,1])    # bevfusion case
        self.do_custom_test([38153], [1180, 180, 5], 128, 256, [3,3,3], 1, [1,1,2], [0,0,1])    # bevfusion case
        self.do_custom_test([38153], [1180, 180, 5], 128, 256, [5,5,5], 1, [2,2,2], [1,1,1])    # K = 5
        self.do_custom_test([23787], [3571, 4251, 1062], 4, 32, [3,3,3], 1, [2,2,2], [1,1,1])    # test large spatial shape
        self.do_custom_test([50000], [128, 128, 128], 1024, 1024, [3,3,3], 1, [2,2,2], [1,1,1])    # 1024 channel
        self.do_custom_test([370000], [1440, 1440, 41], 16, 32, [3,3,3], 1, [2,2,2], [1,1,1])    # large points

if __name__ == "__main__":
    np.random.seed(100)	
    run_tests()