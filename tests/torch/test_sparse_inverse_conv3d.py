# Copyright (c) 2025, Huawei Technologies.All rights reserved.

import torch, torch_npu
import numpy as np
from mx_driving.spconv import SparseConvTensor, SparseConv3d, SparseInverseConv3d
import spconv.pytorch as spconv
from torch import nn
from torch_npu.testing.testcase import TestCase, run_tests


def generate_sparse_data(
    shape, num_points, num_channels, integer=False, data_range=(-1, 1), dtype=np.float32, shape_scale=1
):
    dense_shape = shape
    ndim = len(dense_shape)
    num_points = np.array(num_points)
    batch_size = len(num_points)
    batch_indices = []
    coors_total = np.stack(np.meshgrid(*[np.arange(0, s // shape_scale) for s in shape]), axis=-1)
    coors_total = coors_total.reshape(-1, ndim) * shape_scale
    for i in range(batch_size):
        np.random.shuffle(coors_total)
        inds_total = coors_total[: num_points[i]]
        inds_total = np.pad(inds_total, ((0, 0), (0, 1)), mode="constant", constant_values=i)
        batch_indices.append(inds_total)
    if integer:
        sparse_data = np.random.randint(data_range[0], data_range[1], size=[num_points.sum(), num_channels]).astype(
            dtype
        )
    else:
        sparse_data = np.random.uniform(data_range[0], data_range[1], size=[num_points.sum(), num_channels]).astype(
            dtype
        )

    res = {
        "features": sparse_data.astype(dtype),
    }
    batch_indices = np.concatenate(batch_indices, axis=0)
    res["indices"] = batch_indices.astype(np.int32)
    return res


class CustomNet(nn.Module):
    def __init__(
        self,
        spatial_shape,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        indice_key=None,
        mode="spconv",
    ):
        super().__init__()
        self.conv3d = SparseConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            indice_key=indice_key,
            mode=mode,
        )
        self.inv_conv3d = SparseInverseConv3d(
            in_channels=out_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            indice_key=indice_key,
            mode=mode,
        )
        self.spatial_shape = spatial_shape

    def forward(self, features, coors, batch_size):
        x = SparseConvTensor(features, coors, self.spatial_shape, batch_size)
        x1 = self.conv3d(x)
        x2 = self.inv_conv3d(x1)
        return x2


class SpconvNet(nn.Module):
    def __init__(
        self,
        spatial_shape,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        indice_key=None,
    ):
        super().__init__()
        self.conv3d = spconv.SparseConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            indice_key=indice_key,
        )
        self.inv_conv3d = spconv.SparseInverseConv3d(
            in_channels=out_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            bias = bias,
            indice_key=indice_key,
        )
        self.spatial_shape = spatial_shape

    def forward(self, features, coors, batch_size):
        x = spconv.SparseConvTensor(
            features=features, indices=coors, spatial_shape=self.spatial_shape, batch_size=batch_size
        )
        x1 = self.conv3d(x)
        x2 = self.inv_conv3d(x1)
        return x2


def getout(
    spatial_shape,
    feature_num,
    batch_size,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    bias,
    dtype,
    indice_key=None,
):
    sparse_dict = generate_sparse_data(spatial_shape, [feature_num] * batch_size, in_channels, dtype=np.float32)
    voxels = np.ascontiguousarray(sparse_dict["features"]).astype(np.float32)
    coors = np.ascontiguousarray(sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)

    voxels_th = torch.from_numpy(voxels).to(dtype)
    coors_th = torch.from_numpy(coors)

    # CustomNet
    custom_net = (
        CustomNet(
            spatial_shape, in_channels, out_channels, kernel_size, stride, padding, dilation, bias, indice_key, "spconv"
        )
        .npu()
        .to(dtype)
    )

    forward_weight_shape = custom_net.conv3d.weight.shape
    inverse_weight_shape = custom_net.inv_conv3d.weight.shape

    forward_weight = torch.rand(forward_weight_shape, dtype=dtype)
    inverse_weight = torch.rand(inverse_weight_shape, dtype=dtype)

    custom_net.conv3d.weight.data = forward_weight.npu()
    custom_net.inv_conv3d.weight.data = inverse_weight.npu()

    custom_output = custom_net(voxels_th.npu(), coors_th.npu(), batch_size)

    # SpconvNet
    spconv_net = SpconvNet(
        spatial_shape, in_channels, out_channels, kernel_size, stride, padding, dilation, bias, indice_key
    ).to(dtype)

    # d h w in out -> d h w out in
    spconv_net.conv3d.weight.data = forward_weight.permute(0, 1, 2, 4, 3).contiguous()
    spconv_net.inv_conv3d.weight.data = inverse_weight.permute(0, 1, 2, 4, 3).contiguous()

    spconv_output = spconv_net(voxels_th, coors_th, batch_size)

    return custom_output.features.flatten(), spconv_output.features.flatten().npu()


class TestSparseConv3d(TestCase):
    def test_model_case0(self):
        feature_num = 1529
        batch_size = 4
        spatial_shape = [80, 160, 160]
        in_channels = 192
        out_channels = 256
        kernel_size = 3
        stride = 2
        padding = 0
        dilation = 1
        bias = False
        dtype = torch.float32
        indice_key = "test"

        custom_out, spconv_out = getout(
            spatial_shape,
            feature_num,
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias,
            dtype,
            indice_key,
        )
        self.assertRtolEqual(custom_out, spconv_out, 1e-3, 1e-3)


if __name__ == "__main__":
    np.random.seed(50051)
    torch.manual_seed(50051)
    run_tests()
