# Copyright (c) 2024, Huawei Technologies.All rights reserved.
# Copyright 2019 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np
import torch
from torch.nn import init
from torch.nn.init import calculate_gain
from torch.nn.parameter import Parameter

from . import sparse_functional as Fsp
from . import sparse_ops as ops
from .sparse_modules import SparseModule
from .sparse_structure import SparseConvTensor


def _calculate_fan_in_and_fan_out_hwio(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError('fan in and fan out can not be computed for tensor'
                         'with fewer than 2 dimensions')

    if dimensions == 2:  # Linear
        fan_in = tensor.size(-2)
        fan_out = tensor.size(-1)
    else:
        num_input_fmaps = tensor.size(-2)
        num_output_fmaps = tensor.size(-1)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[..., 0, 0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


class SparseConvolution(SparseModule):

    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def __init__(self,
                 ndim,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 subm=False,
                 output_padding=0,
                 transposed=False,
                 inverse=False,
                 indice_key=None,
                 fused_bn=False,
                 mode='mmcv'):
        super().__init__()
        if groups != 1:
            raise RuntimeError("do not support group == 1")
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size] * ndim
        if not isinstance(stride, (list, tuple)):
            stride = [stride] * ndim
        if not isinstance(padding, (list, tuple)):
            padding = [padding] * ndim
        if not isinstance(dilation, (list, tuple)):
            dilation = [dilation] * ndim
        if not isinstance(output_padding, (list, tuple)):
            output_padding = [output_padding] * ndim

        for d, s in zip(dilation, stride):
            if not any([s == 1, d == 1]):
                raise RuntimeError("do not support s == 1, d == 1")

        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv1x1 = np.prod(kernel_size) == 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.inverse = inverse
        self.output_padding = output_padding
        self.groups = groups
        self.subm = subm
        self.indice_key = indice_key
        self.fused_bn = fused_bn
        self.mode = mode

        self.weight = Parameter(
            torch.Tensor(*kernel_size, in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        fan_in, fan_out = _calculate_fan_in_and_fan_out_hwio(self.weight)
        if self.mode == 'mmcv':
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            self._custom_kaiming_uniform_(self.weight, a=math.sqrt(5), fan_in=fan_in, fan_out=fan_out)
        if self.bias is not None:
            if fan_in == 0:
                bound = 0
            else:
                bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _custom_kaiming_uniform_(self,
                                 tensor,
                                 a=0,
                                 fan_in=0,
                                 fan_out=0,
                                 mode='fan_in',
                                 nonlinearity='leaky_relu'):
        fan = 0.0
        if mode == 'fan_in':
            fan = float(fan_in)
        elif mode == 'fan_out':
            fan = float(fan_out)
        gain = calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
            tensor.data = tensor.data.reshape(self.out_channels, np.prod(self.kernel_size) * self.in_channels).transpose(-1, -2).contiguous()
            tensor.data = tensor.data.reshape(*self.kernel_size, self.in_channels, self.out_channels)

    def forward(self, input):
        if not isinstance(input, SparseConvTensor):
            raise RuntimeError("input is not SparseConvTensor")
        if self.inverse:
            out_spatial_shape = ops.get_inverse_conv_output_size(
                input.spatial_shape, self.kernel_size, self.stride, self.padding, self.dilation, self.output_padding)
            out_spatial_shape = [int(i) for i in out_spatial_shape]
            if not isinstance(out_spatial_shape, list):
                out_spatial_shape = out_spatial_shape.tolist()
            out_features, outidx = Fsp.indice_inverse_conv(input.features, input.indices, self.weight, out_spatial_shape,
                                                           self.out_channels, input.batch_size,
                                                           self.kernel_size, self.stride, self.padding, self.dilation, self.output_padding,
                                                           self.groups, self.bias)
        elif not self.subm:
            out_spatial_shape = ops.get_conv_output_size(
                input.spatial_shape, self.kernel_size, self.stride, self.padding, self.dilation)
            out_spatial_shape = [int(i) for i in out_spatial_shape]
            if not isinstance(out_spatial_shape, list):
                out_spatial_shape = out_spatial_shape.tolist()
            out_features, outidx = Fsp.indice_conv(input.features, input.indices, self.weight, out_spatial_shape,
                                                   self.out_channels, input.batch_size,
                                                   self.kernel_size, self.stride, self.padding, self.dilation,
                                                   self.groups, self.bias)
        else:
            out_spatial_shape = input.spatial_shape
            out_spatial_shape = [int(i) for i in out_spatial_shape]
            if not isinstance(out_spatial_shape, list):
                out_spatial_shape = out_spatial_shape.tolist()
            out_features, outidx = Fsp.indice_subm_conv(input.features, input.indices, self.weight, out_spatial_shape,
                                                        self.out_channels, input.batch_size,
                                                        self.kernel_size, self.stride, self.padding, self.dilation,
                                                        self.groups, self.bias)

        if self.bias is not None:
            out_features += self.bias

        out_tensor = SparseConvTensor(out_features, outidx, out_spatial_shape,
                                        input.batch_size)
        return out_tensor


class SparseConv3d(SparseConvolution):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None,
                 mode='mmcv'):
        super().__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            indice_key=indice_key,
            mode=mode)


class SubMConv3d(SparseConvolution):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 indice_key=None,
                 mode='mmcv'):
        super().__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            True,
            indice_key=indice_key,
            mode=mode)


class SparseInverseConv3d(SparseConvolution):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 inverse=True,
                 indice_key=None,
                 mode='mmcv'):
        super().__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            subm=False,
            inverse=True,
            indice_key=indice_key,
            mode=mode)