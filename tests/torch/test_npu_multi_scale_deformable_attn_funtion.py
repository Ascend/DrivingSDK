import unittest
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import ads.common


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def multi_scale_deformable_attn_pytorch( 
        value: torch.Tensor, value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor) -> torch.Tensor:
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ =\
        sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)

        sampling_grid_l_ = sampling_grids[:, :, :,
                                          level].transpose(1, 2).flatten(0, 1)

        sampling_value_l_ = torch.nn.functional.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)

    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    return output.transpose(1, 2).contiguous()


class TestMultiScaleDeformableAttnFunction(TestCase):
    def gen_data(self, shape, dtype):
        bs, num_heads, embed_dims, num_levels, num_points, num_queries = shape
        cpu_shapes = torch.tensor([6, 4] * num_levels).reshape(num_levels, 2)
        num_keys = sum((H * W).item() for H, W in cpu_shapes)

        cpu_value = torch.rand(bs, num_keys, num_heads, embed_dims) * 0.01
        cpu_sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2)
        cpu_attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points) + 1e-5

        cpu_offset = torch.cat((cpu_shapes.new_zeros((1, )), cpu_shapes.prod(1).cumsum(0)[:-1]))
 
        npu_value = cpu_value.npu()
        npu_shapes = cpu_shapes.npu()
        npu_offset = cpu_offset.npu()
        npu_sampling_locations = cpu_sampling_locations.npu()
        npu_attention_weights = cpu_attention_weights.npu()
        
        return [cpu_value, cpu_shapes, cpu_offset, cpu_sampling_locations, cpu_attention_weights], [npu_value, npu_shapes, npu_offset, npu_sampling_locations, npu_attention_weights]

    def cpu_to_exec(self, cpu_data):
        output = multi_scale_deformable_attn_pytorch(cpu_data[0].double(), cpu_data[1].long(), cpu_data[3].double(), cpu_data[4].double())
        return output.float().numpy()

    def npu_to_exec(self, npu_data):
        output = ads.common.npu_multi_scale_deformable_attn_function(npu_data[0], npu_data[1], npu_data[2], npu_data[3], npu_data[4])
        return output.cpu().numpy()

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `MultiScaleDeformableAttnFunction` is only supported on 910B, skip this ut!")
    def test_multi_scale_deformable_attn_function(self):
        dtype_list = [torch.float32]
        shape_list = [
            [6, 8, 32, 4, 8, 9680], [3, 4, 16, 6, 3, 7], [1, 8, 32, 4, 8, 30832]
        ]
        items = [
            [shape, dtype]
            for shape in shape_list
            for dtype in dtype_list
        ]
        for shape, dtype in items:
            cpu_x, npu_x = self.gen_data(shape, dtype)
            cpu_out = self.cpu_to_exec(cpu_x)
            npu_out = self.npu_to_exec(npu_x)
            self.assertRtolEqual(cpu_out, npu_out)


if __name__ == '__main__':
    run_tests()
