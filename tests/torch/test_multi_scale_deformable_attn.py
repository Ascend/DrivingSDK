import unittest
from collections import namedtuple

import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving


# pylint: disable=too-many-return-values
@golden_data_cache(__file__)
def cpu_gen_inputs(shape):
    bs, num_queries, embed_dims, num_heads, num_levels, num_points = shape
    shapes = torch.tensor([60, 40] * num_levels).reshape(num_levels, 2)
    num_keys = sum((H * W).item() for H, W in shapes)

    value = torch.rand(bs, num_keys, num_heads, embed_dims) * 0.01
    sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2)
    attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points) + 1e-5
    offset = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
    grad_output = torch.rand(bs, num_queries, num_heads * embed_dims) * 1e-3

    return shapes, num_keys, value, sampling_locations, attention_weights, offset, grad_output


@golden_data_cache(__file__)
def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)

        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)

        sampling_value_l_ = torch.nn.functional.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)

    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


@golden_data_cache(__file__)
def multi_scale_deformable_attn_pytorch_grad(
    cpu_output, cpu_grad_output, cpu_value, cpu_sampling_locations, cpu_attention_weights
):
    cpu_output.backward(cpu_grad_output)
    grad_value = cpu_value.grad.float().numpy()
    grad_sampling_locations = cpu_sampling_locations.grad.float().numpy()
    grad_attention_weights = cpu_attention_weights.grad.float().numpy()

    return grad_value, grad_sampling_locations, grad_attention_weights


ExecResults = namedtuple("ExecResults", ["output", "grad_value", "grad_sampling_locations", "grad_attention_weights"])
Inputs = namedtuple("Inputs", ["value", "shapes", "offset", "sampling_locations", "attention_weights", "grad_output"])


class TestMultiScaleDeformableAttnFunction(TestCase):
    def gen_inputs(self, shape, dtype):
        bs, num_queries, embed_dims, num_heads, num_levels, num_points = shape
        shapes, num_keys, value, sampling_locations, attention_weights, offset, grad_output = cpu_gen_inputs(shape)

        cpu_value = value.double()
        cpu_shapes = shapes.long()
        cpu_sampling_locations = sampling_locations.double()
        cpu_attention_weights = attention_weights.double()
        cpu_grad_output = grad_output.double()

        cpu_value.requires_grad_()
        cpu_sampling_locations.requires_grad_()
        cpu_attention_weights.requires_grad_()

        npu_value = value.npu()
        npu_shapes = shapes.npu()
        npu_offset = offset.npu()
        npu_sampling_locations = sampling_locations.npu()
        npu_attention_weights = attention_weights.npu()
        npu_grad_output = grad_output.npu()

        npu_value.requires_grad_()
        npu_sampling_locations.requires_grad_()
        npu_attention_weights.requires_grad_()

        return Inputs(
            cpu_value, cpu_shapes, None, cpu_sampling_locations, cpu_attention_weights, cpu_grad_output
        ), Inputs(npu_value, npu_shapes, npu_offset, npu_sampling_locations, npu_attention_weights, npu_grad_output)

    def cpu_to_exec(self, cpu_inputs):
        cpu_value = cpu_inputs.value
        cpu_shapes = cpu_inputs.shapes
        cpu_sampling_locations = cpu_inputs.sampling_locations
        cpu_attention_weights = cpu_inputs.attention_weights
        cpu_grad_output = cpu_inputs.grad_output
        cpu_output = multi_scale_deformable_attn_pytorch(
            cpu_value, cpu_shapes, cpu_sampling_locations, cpu_attention_weights
        )
        grad_value, grad_sampling_locations, grad_attention_weights = multi_scale_deformable_attn_pytorch_grad(
            cpu_output, cpu_grad_output, cpu_value, cpu_sampling_locations, cpu_attention_weights
        )
        return ExecResults(
            output=cpu_output.detach().float().numpy(),
            grad_value=grad_value,
            grad_sampling_locations=grad_sampling_locations,
            grad_attention_weights=grad_attention_weights,
        )

    def npu_to_exec(self, npu_inputs):
        npu_value = npu_inputs.value
        npu_shapes = npu_inputs.shapes
        npu_offset = npu_inputs.offset
        npu_sampling_locations = npu_inputs.sampling_locations
        npu_attention_weights = npu_inputs.attention_weights
        npu_grad_output = npu_inputs.grad_output
        npu_output = mx_driving.multi_scale_deformable_attn(
            npu_value, npu_shapes, npu_offset, npu_sampling_locations, npu_attention_weights
        )
        npu_output.backward(npu_grad_output)
        return ExecResults(
            output=npu_output.detach().cpu().numpy(),
            grad_value=npu_value.grad.cpu().numpy(),
            grad_sampling_locations=npu_sampling_locations.grad.cpu().numpy(),
            grad_attention_weights=npu_attention_weights.grad.cpu().numpy(),
        )

    # fast_mode: num_heads * num_points * num_levels <= 64
    def test_fast_mode(self):
        shape = [6, 9680, 32, 8, 1, 8]
        cpu_inputs, npu_inputs = self.gen_inputs(shape, torch.float32)
        cpu_results = self.cpu_to_exec(cpu_inputs)
        npu_results = self.npu_to_exec(npu_inputs)
        self.assertRtolEqual(cpu_results.output, npu_results.output)
        self.assertRtolEqual(cpu_results.grad_value, npu_results.grad_value)
        self.assertRtolEqual(cpu_results.grad_attention_weights, npu_results.grad_attention_weights)
        self.assertRtolEqual(cpu_results.grad_sampling_locations, npu_results.grad_sampling_locations)

    def test_embed_32(self):
        shape = [6, 9680, 32, 8, 4, 4]
        cpu_inputs, npu_inputs = self.gen_inputs(shape, torch.float32)
        cpu_results = self.cpu_to_exec(cpu_inputs)
        npu_results = self.npu_to_exec(npu_inputs)
        self.assertRtolEqual(cpu_results.output, npu_results.output)
        self.assertRtolEqual(cpu_results.grad_value, npu_results.grad_value)
        self.assertRtolEqual(cpu_results.grad_attention_weights, npu_results.grad_attention_weights)
        self.assertRtolEqual(cpu_results.grad_sampling_locations, npu_results.grad_sampling_locations)

    def test_embed_unaligned(self):
        shape = [6, 9680, 37, 4, 5, 3]
        cpu_inputs, npu_inputs = self.gen_inputs(shape, torch.float32)
        cpu_results = self.cpu_to_exec(cpu_inputs)
        npu_results = self.npu_to_exec(npu_inputs)
        self.assertRtolEqual(cpu_results.output, npu_results.output)
        self.assertRtolEqual(cpu_results.grad_value, npu_results.grad_value)
        self.assertRtolEqual(cpu_results.grad_attention_weights, npu_results.grad_attention_weights)
        self.assertRtolEqual(cpu_results.grad_sampling_locations, npu_results.grad_sampling_locations)

    def test_embed_16(self):
        shape = [1, 27216, 16, 5, 3, 1]
        cpu_inputs, npu_inputs = self.gen_inputs(shape, torch.float32)
        cpu_results = self.cpu_to_exec(cpu_inputs)
        npu_results = self.npu_to_exec(npu_inputs)
        self.assertRtolEqual(cpu_results.output, npu_results.output)
        self.assertRtolEqual(cpu_results.grad_value, npu_results.grad_value)
        self.assertRtolEqual(cpu_results.grad_attention_weights, npu_results.grad_attention_weights)
        self.assertRtolEqual(cpu_results.grad_sampling_locations, npu_results.grad_sampling_locations)

    def test_embed_64(self):
        shape = [1, 1450, 64, 6, 1, 2]
        cpu_inputs, npu_inputs = self.gen_inputs(shape, torch.float32)
        cpu_results = self.cpu_to_exec(cpu_inputs)
        npu_results = self.npu_to_exec(npu_inputs)
        self.assertRtolEqual(cpu_results.output, npu_results.output)
        self.assertRtolEqual(cpu_results.grad_value, npu_results.grad_value)
        self.assertRtolEqual(cpu_results.grad_attention_weights, npu_results.grad_attention_weights)
        self.assertRtolEqual(cpu_results.grad_sampling_locations, npu_results.grad_sampling_locations)

    def test_fully_embed_64(self):
        shape = [1, 1450, 64, 7, 8, 8]
        cpu_inputs, npu_inputs = self.gen_inputs(shape, torch.float32)
        cpu_results = self.cpu_to_exec(cpu_inputs)
        npu_results = self.npu_to_exec(npu_inputs)
        self.assertRtolEqual(cpu_results.output, npu_results.output)
        self.assertRtolEqual(cpu_results.grad_value, npu_results.grad_value)
        self.assertRtolEqual(cpu_results.grad_attention_weights, npu_results.grad_attention_weights)
        self.assertRtolEqual(cpu_results.grad_sampling_locations, npu_results.grad_sampling_locations)

    def test_point_16(self):
        shape = [1, 1890, 32, 7, 4, 16]
        cpu_inputs, npu_inputs = self.gen_inputs(shape, torch.float32)
        cpu_results = self.cpu_to_exec(cpu_inputs)
        npu_results = self.npu_to_exec(npu_inputs)
        self.assertRtolEqual(cpu_results.output, npu_results.output)
        self.assertRtolEqual(cpu_results.grad_value, npu_results.grad_value)
        self.assertRtolEqual(cpu_results.grad_attention_weights, npu_results.grad_attention_weights)
        self.assertRtolEqual(cpu_results.grad_sampling_locations, npu_results.grad_sampling_locations)


if __name__ == "__main__":
    run_tests()
