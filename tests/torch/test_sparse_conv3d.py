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
from mx_driving.spconv import SparseSequential, SparseConvTensor, SparseConv3d


def generate_sparse_data(shape,
                        num_points,
                        num_channels,
                        integer = False,
                        data_range = (-1, 1),
                        with_dense = True,
                        dtype = np.float32,
                        shape_scale = 1):
    dense_shape = shape
    ndim = len(dense_shape)
    # num_points = np.random.randint(10, 100, size=[batch_size, ndim])
    num_points = np.array(num_points)
    # num_points = np.array([3, 2])
    batch_size = len(num_points)
    batch_indices = []
    coors_total = np.stack(np.meshgrid(*[np.arange(0, s // shape_scale) for s in shape]),
                           axis=-1)
    coors_total = coors_total.reshape(-1, ndim) * shape_scale
    for i in range(batch_size):
        np.random.shuffle(coors_total)
        inds_total = coors_total[:num_points[i]]
        inds_total = np.pad(inds_total, ((0, 0), (0, 1)),
                            mode="constant",
                            constant_values=i)
        batch_indices.append(inds_total)
    if integer:
        sparse_data = np.random.randint(data_range[0],
                                        data_range[1],
                                        size=[num_points.sum(),
                                              num_channels]).astype(dtype)
    else:
        sparse_data = np.random.uniform(data_range[0],
                                        data_range[1],
                                        size=[num_points.sum(),
                                              num_channels]).astype(dtype)

    # sparse_data = np.arange(1, num_points.sum() + 1).astype(np.float32).reshape(5, 1)

    res = {
        "features": sparse_data.astype(dtype),
    }
    if with_dense:
        dense_data = np.zeros([batch_size, num_channels, *dense_shape],
                              dtype=sparse_data.dtype)
        start = 0
        for i, inds in enumerate(batch_indices):
            for j, ind in enumerate(inds):
                dense_slice = (i, slice(None), *ind[:-1])
                dense_data[dense_slice] = sparse_data[start + j]
            start += len(inds)
        res["features_dense"] = dense_data.astype(dtype)
    batch_indices = np.concatenate(batch_indices, axis=0)
    res["indices"] = batch_indices.astype(np.int32)
    return res


class Net(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.net = SparseSequential(
            SparseConv3d(16, 32, 3)
        )
        max_batch_size = 1
        self.shape = shape

    def forward(self, features, coors, batch_size):
        x = SparseConvTensor(features,
                            coors,
                            self.shape,
                            batch_size)
        return self.net(x)


def _test_multi_impl(spatial_shape, feature_num, dtype: torch.dtype):

    np.random.seed(50051)

    spatial_shape = [4, 4, 4]
    sparse_dict = generate_sparse_data(spatial_shape, [feature_num] * 1, 16)

    voxels = np.ascontiguousarray(sparse_dict["features"]).astype(np.float32)
    coors = np.ascontiguousarray(
        sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
    device = torch.device("npu:0")

    voxels_th_npu = torch.from_numpy(voxels).to(device).to(dtype)

    coors_th_npu = torch.from_numpy(coors).to(device)
    net_cls = Net
    # npu
    torch.manual_seed(50051)
    net_native_npu = net_cls(spatial_shape).to(device).to(dtype)

    out = net_native_npu(voxels_th_npu, coors_th_npu, 1)


def test_multi_impl():
    _test_multi_impl([4, 4, 4], 3, torch.float32)
    _test_multi_impl([7, 7, 7], 9, torch.float32)
    _test_multi_impl([12, 13, 14], 100, torch.float32)
    _test_multi_impl([25, 25, 25], 400, torch.float32)


if __name__ == "__main__":
    test_multi_impl()