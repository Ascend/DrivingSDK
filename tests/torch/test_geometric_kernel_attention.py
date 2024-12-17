"""
Copyright (c) 2022 Hust Vision Lab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Licensed under the MIT License.
"""
import unittest
from collections import namedtuple

import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


# pylint: disable=too-many-return-values
@golden_data_cache(__file__)
def cpu_gen_inputs(shape):
    bs, num_queries, embed_dims, num_heads, num_levels, num_points = shape
    
    sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2)
    if bs == 24:
        spatial_shapes = torch.tensor([15, 25] * num_levels).reshape(num_levels, 2)
        sampling_locations[:, :, :, :, :, 0] = sampling_locations[:, :, :, :, :, 0] * 21 - 3 # -3 ~ 18
        sampling_locations[:, :, :, :, :, 1] = sampling_locations[:, :, :, :, :, 1] * 31 - 3 # -3 ~ 28
    else:
        spatial_shapes = torch.tensor([6, 10] * num_levels).reshape(num_levels, 2)
        sampling_locations[:, :, :, :, :, 0] = sampling_locations[:, :, :, :, :, 0] * 12 - 3 # -3 ~ 9
        sampling_locations[:, :, :, :, :, 1] = sampling_locations[:, :, :, :, :, 1] * 16 - 3 # -3 ~ 13
    num_keys = sum((H * W).item() for H, W in spatial_shapes)

    value = torch.rand(bs, num_keys, num_heads, embed_dims) * 3
    attn_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points) * 1
    level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
    grad_output = torch.rand(bs, num_queries, num_heads * embed_dims) * 10

    return sampling_locations, spatial_shapes, value, attn_weights, level_start_index, grad_output


@golden_data_cache(__file__)
def cpu_geometric_kernel_attention(value, spatial_shapes, level_start_index, sampling_locations, attn_weights):
    """CPU version of geometric kernel attention.

    Args:
        value (Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims)
        spatial_shapes (Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last embed_dimsension 2 represent (h, w)
        sampling_locations (Tensor): The location of sampling points,
            has shape
            (bs, num_queries, num_heads, num_levels, num_points, 2),
            the last embed_dimsension 2 represent (x, y).
        attn_weights (Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs, num_queries, num_heads, num_levels, num_points),

    Returns:
        Tensor: has shape (bs, num_queries, num_heads * embed_dims)
    """
    bs, num_keys, num_heads, embed_dims = value.shape

    value = value.transpose(1, 2).contiguous().view(bs * num_heads * num_keys, embed_dims)
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    
    with torch.no_grad():
        sampling_index = sampling_locations.new_zeros(
            (bs, num_queries, num_heads, num_levels, num_points)).to(value.device)

        for level, (H, W) in enumerate(spatial_shapes):
            sampling_locations[:, :, :, level, :, 0].clamp_(min=0, max=W - 1)
            sampling_locations[:, :, :, level, :, 1].clamp_(min=0, max=H - 1)
            sampling_index[:, :, :, level] = level_start_index[level] + \
                                             sampling_locations[:, :, :, level, :, 0] + \
                                             sampling_locations[:, :, :, level, :, 1] * W

        sampling_index = sampling_index.transpose(1, 2).reshape(bs, num_heads, -1)
        sampling_index = sampling_index + \
            (torch.arange(num_heads).to(sampling_index) * num_keys).view(1, num_heads, 1)
        sampling_index = sampling_index.reshape(bs, -1) + \
            (torch.arange(bs).to(sampling_index) * num_keys * num_heads).view(bs, 1)

    sampling_value = value[sampling_index].view(
        bs, num_heads, num_queries, num_levels * num_points, embed_dims)
    attn_weights = attn_weights.transpose(1, 2).contiguous().view(
        bs, num_heads, num_queries, num_levels * num_points, 1)

    output = (sampling_value * attn_weights).sum(-2).transpose(1, 2).contiguous()
    return output.view(bs, num_queries, -1)


@golden_data_cache(__file__)
def cpu_geometric_kernel_attention_grad(cpu_output, grad_output, value, attn_weights):
    cpu_output.backward(grad_output)

    grad_value = value.grad.float().numpy()
    grad_attn_weights = attn_weights.grad.float().numpy()

    return grad_value, grad_attn_weights


ExecResults = namedtuple('ExecResults', ['output', 'grad_value', 'grad_attn_weights'])
Inputs = namedtuple('Inputs', ['value', 'spatial_shapes', 'level_start_index', 'sampling_locations', 'attn_weights', 'grad_output'])


class TestGeometricKernelAttention(TestCase):
    def setUp(self):
        self.dtype_list = [torch.float32]
        self.shape_list = [
            [24, 7156, 64, 4, 1, 15], [24, 7129, 64, 4, 1, 15], 
            [24, 7203, 64, 4, 1, 15], [24, 7173, 64, 4, 1, 15], 
            [144, 1146, 64, 4, 1, 15], [144, 1136, 64, 4, 1, 15], 
            [144, 1152, 64, 4, 1, 15], [2, 4, 16, 4, 2, 15]
        ]

        self.items = [[shape, dtype] for shape in self.shape_list for dtype in self.dtype_list]
        self.test_results = self.gen_results()

    def gen_results(self):
        test_results = []
        for shape, dtype in self.items:
            cpu_inputs, npu_inputs = self.gen_inputs(shape, dtype)
            cpu_results = self.cpu_to_exec(cpu_inputs)
            npu_results = self.npu_to_exec(npu_inputs)
            test_results.append((cpu_results, npu_results))
        return test_results

    def gen_inputs(self, shape, dtype):

        sampling_locations, spatial_shapes, value, attn_weights, level_start_index, grad_output = cpu_gen_inputs(shape)
        
        cpu_value = value.float()
        cpu_spatial_shapes = spatial_shapes.long()
        cpu_level_start_index = level_start_index.long()
        cpu_sampling_locations = sampling_locations.long()
        cpu_attn_weights = attn_weights.float()
        cpu_grad_output = grad_output.float()

        npu_value = cpu_value.float().npu()
        npu_spatial_shapes = cpu_spatial_shapes.int().npu()
        npu_level_start_index = cpu_level_start_index.int().npu()
        npu_sampling_locations = cpu_sampling_locations.float().npu()
        npu_attn_weights = cpu_attn_weights.float().npu()
        npu_grad_output = cpu_grad_output.float().npu()
        
        cpu_value.requires_grad_()
        cpu_attn_weights.requires_grad_()
        
        npu_value.requires_grad_()
        npu_attn_weights.requires_grad_()
        
        return Inputs(cpu_value, cpu_spatial_shapes, cpu_level_start_index, cpu_sampling_locations, cpu_attn_weights, cpu_grad_output), \
               Inputs(npu_value, npu_spatial_shapes, npu_level_start_index, npu_sampling_locations, npu_attn_weights, npu_grad_output)

    def cpu_to_exec(self, cpu_inputs):
        value = cpu_inputs.value
        spatial_shapes = cpu_inputs.spatial_shapes
        level_start_index = cpu_inputs.level_start_index
        sampling_locations = cpu_inputs.sampling_locations
        attn_weights = cpu_inputs.attn_weights
        grad_output = cpu_inputs.grad_output
        cpu_output = cpu_geometric_kernel_attention(
            value, spatial_shapes, level_start_index, sampling_locations, attn_weights
        )
        grad_value, grad_attn_weights = cpu_geometric_kernel_attention_grad(
            cpu_output, grad_output, value, attn_weights
        )
        
        return ExecResults(
            output=cpu_output.detach().float().numpy(),
            grad_value=grad_value,
            grad_attn_weights=grad_attn_weights
        )

    def npu_to_exec(self, npu_inputs):
        value = npu_inputs.value
        spatial_shapes = npu_inputs.spatial_shapes
        level_start_index = npu_inputs.level_start_index
        sampling_locations = npu_inputs.sampling_locations
        attn_weights = npu_inputs.attn_weights
        grad_output = npu_inputs.grad_output
        npu_output = mx_driving.geometric_kernel_attention(
            value, spatial_shapes, level_start_index, sampling_locations, attn_weights
        )
        npu_output.backward(grad_output)
        return ExecResults(
            output=npu_output.detach().cpu().numpy(),
            grad_value=value.grad.cpu().numpy(),
            grad_attn_weights=attn_weights.grad.cpu().numpy()
        )

    def test_geometric_kernel_attention_forward(self):
        for cpu_results, npu_results in self.test_results:
            self.assertRtolEqual(cpu_results.output, npu_results.output)
            
    def test_geometric_kernel_attention_backward(self):
        for cpu_results, npu_results in self.test_results:
            self.assertRtolEqual(cpu_results.grad_value, npu_results.grad_value)
            self.assertRtolEqual(cpu_results.grad_attn_weights, npu_results.grad_attn_weights)


if __name__ == "__main__":
    run_tests()
