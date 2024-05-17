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


class TestMultiScaleDeformableAttentionV2Grad(TestCase):
    def gen_data(self, shape, dtype):
        bs, num_queries, embed_dims, num_heads, num_levels, num_points = shape
        shapes = torch.tensor([6, 4] * num_levels).reshape(num_levels, 2)
        num_keys = sum((H * W).item() for H, W in shapes)

        value = torch.rand(bs, num_keys, num_heads, embed_dims) * 0.01
        sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2)
        attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points) + 1e-5
        offset = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))
        grad_output = torch.rand(bs, num_queries, num_heads * embed_dims)
        
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
        
        return [cpu_value, cpu_shapes, cpu_sampling_locations, cpu_attention_weights, cpu_grad_output], [npu_value, npu_shapes, npu_offset, npu_sampling_locations, npu_attention_weights, npu_grad_output]

    def cpu_to_exec(self, cpu_data):
        cpu_value = cpu_data[0]
        cpu_shapes = cpu_data[1]
        cpu_sampling_locations = cpu_data[2]
        cpu_attention_weights = cpu_data[3]
        cpu_grad_output = cpu_data[4]
        output = multi_scale_deformable_attn_pytorch(cpu_value, cpu_shapes, cpu_sampling_locations, cpu_attention_weights)
        output.backward(cpu_grad_output)
        return cpu_value.grad.float().numpy(), cpu_sampling_locations.grad.float().numpy(), cpu_attention_weights.grad.float().numpy()

    def npu_to_exec(self, npu_data):
        npu_value = npu_data[0]
        npu_shapes = npu_data[1]
        npu_offset = npu_data[2]
        npu_sampling_locations = npu_data[3]
        npu_attention_weights = npu_data[4]
        npu_grad_output = npu_data[5]
        output = multi_scale_deformable_attn_pytorch(npu_value, npu_shapes, npu_sampling_locations, npu_attention_weights)
        output.backward(npu_grad_output)
        return npu_value.grad.cpu().numpy(), npu_sampling_locations.grad.cpu().numpy(), npu_attention_weights.grad.cpu().numpy()

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `MultiScaleDeformableAttentionV2Grad` is only supported on 910B, skip this ut!")
    def test_multi_scale_deformable_attention_v2_grad(self):
        dtype_list = [torch.float32]
        shape_list = [
            [6, 9680, 32, 8, 4, 4], [1, 30832, 32, 8, 4, 4], [1, 36864, 32, 8, 4, 4], 
            [1, 27216, 32, 8, 3, 4], [1, 500, 32, 8, 3, 4], [2, 10191, 32, 8, 4, 4], 
            [6, 50012, 16, 4, 4, 4], [1, 188232, 32, 8, 5, 4], [1, 1890, 32, 8, 5, 4], [1, 40000, 256, 8, 1, 4]
        ]
        items = [
            [shape, dtype]
            for shape in shape_list
            for dtype in dtype_list
        ]
        for shape, dtype in items:
            cpu_x, npu_x = self.gen_data(shape, dtype)
            cpu_grad_value, cpu_grad_sampling_locations, cpu_grad_attention_weights = self.cpu_to_exec(cpu_x)
            npu_grad_value, npu_grad_sampling_locations, npu_grad_attention_weights = self.npu_to_exec(npu_x)
            self.assertRtolEqual(cpu_grad_value, npu_grad_value)

 
if __name__ == '__main__':
    run_tests()
