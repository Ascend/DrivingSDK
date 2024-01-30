import unittest
import torch
import numpy as np
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import ads.common

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def cal_grad_value(num_point, channels, v1_ub, offset1, offset2, base_ptr, value, value_ptr_offset, grad_h_weight_ub,
                   grad_w_weight_ub, h_w_w1, h_w_w2, grad_value, top_grad_value_ub, w_weight, cp, neg_w1, neg_w2):
    ptr = offset1[cp] + offset2[cp] + base_ptr
    v1_ub[:] = value[value_ptr_offset + ptr: value_ptr_offset + ptr + channels]
    if neg_w1:
        grad_h_weight_ub[cp,:] -= v1_ub * h_w_w1[cp]
    else:
        grad_h_weight_ub[cp,:] += v1_ub * h_w_w1[cp]
    if neg_w2:
        grad_w_weight_ub[cp,:] -= v1_ub * h_w_w2[cp]
    else:
        grad_w_weight_ub[cp,:] += v1_ub * h_w_w2[cp]
    mid_ub = top_grad_value_ub[cp, :] * w_weight[cp]
    grad_value[value_ptr_offset + ptr: value_ptr_offset + ptr + channels] += mid_ub


def ms_scale_deform_att_grad(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output,
                             grad_value, grad_sampling_loc_out, grad_attn_weight):

    batch_size = value.shape[0]
    spatial_size = value.shape[1]
    num_heads = value.shape[2]
    channels = value.shape[3]
    num_levels = spatial_shapes.shape[0]
    num_query = sampling_loc.shape[1]
    num_point = sampling_loc.shape[5]
    print(batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point)
    value = value.flatten()
    grad_attn_weight = grad_attn_weight.flatten()
    grad_sampling_loc_out = grad_sampling_loc_out.flatten()
    grad_value = grad_value.flatten()
    for b in range(batch_size):
        for nq in range(num_query):
            for nh in range(num_heads):
                data_weight_ptr = (b * num_query * num_heads + nq * num_heads + nh) * num_levels * num_point
                data_loc_w_ptr = 2 * data_weight_ptr
                top_grad_ub = grad_output[b, nq, nh, :]
                for nl in range(num_levels):
                    level_start_id = level_start_index[nl]
                    h = spatial_shapes[nl, 0]
                    w = spatial_shapes[nl, 1]
                    qid_stride = num_heads * channels
                    data_value_ptr_init_offset = b * spatial_size * qid_stride
                    value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride
                    loc_w = sampling_loc[b, nq, nh, nl, 0, :]
                    loc_h = sampling_loc[b, nq, nh, nl, 1, :]
                    weight = attn_weight[b, nq, nh, nl, :]
                    h_im = loc_h * h - 0.5
                    w_im = loc_w * w - 0.5
                    h_low = np.floor(h_im).astype(int)
                    w_low = np.floor(w_im).astype(int)
                    h_high = h_low + 1
                    w_high = w_low + 1
                    lh = h_im - h_low
                    lw = w_im - w_low
                    hh = 1 - lh
                    hw = 1 - lw 
                    w_stride = num_heads * channels
                    h_stride = w * w_stride
                    h_low_ptr_offset = h_low * h_stride
                    h_high_ptr_offset = h_low_ptr_offset + h_stride
                    w_low_ptr_offset = w_low * w_stride
                    w_high_ptr_offset = w_low_ptr_offset + w_stride
                    base_ptr = nh * channels
                    w1 = hh * hw
                    w2 = hh * lw
                    w3 = lh * hw
                    w4 = lh * lw 
                    grad_h_weight_ub = np.zeros((num_point, channels))
                    grad_w_weight_ub = np.zeros((num_point, channels))
                    top_grad_value_ub = np.zeros((num_point, channels))
                    grad_weight_ub = np.zeros((num_point, channels))
                    for cp in range(num_point):
                        if h_im[cp] > -1 and w_im[cp] > -1 and h_im[cp] < h and w_im[cp] < w:
                            top_grad_value_ub[cp, :] = weight[cp] * top_grad_ub
                            v1_ub = np.zeros((channels,))
                            if h_low[cp] >= 0 and w_low[cp] >= 0:
                                cal_grad_value(num_point, channels, v1_ub, h_low_ptr_offset, w_low_ptr_offset, base_ptr, value, value_ptr_offset, grad_h_weight_ub,
                                               grad_w_weight_ub, hw, hh, grad_value, top_grad_value_ub, w1, cp, True, True)
                            v2_ub = np.zeros((channels,))
                            if h_low[cp] >= 0 and w_high[cp] < w:
                                cal_grad_value(num_point, channels, v2_ub, h_low_ptr_offset, w_high_ptr_offset, base_ptr, value, value_ptr_offset, grad_h_weight_ub,
                                               grad_w_weight_ub, lw, hh, grad_value, top_grad_value_ub, w2, cp, True, False)
                            v3_ub = np.zeros((channels,))
                            if h_high[cp] < h and w_low[cp] >= 0:
                                cal_grad_value(num_point, channels, v3_ub, h_high_ptr_offset, w_low_ptr_offset, base_ptr, value, value_ptr_offset, grad_h_weight_ub,
                                               grad_w_weight_ub, hw, lh, grad_value, top_grad_value_ub, w3, cp, False, True)
                            v4_ub = np.zeros((channels,))
                            if h_high[cp] < h and w_high[cp] < w:
                                cal_grad_value(num_point, channels, v4_ub, h_high_ptr_offset, w_high_ptr_offset, base_ptr, value, value_ptr_offset, grad_h_weight_ub,
                                               grad_w_weight_ub, lw, lh, grad_value, top_grad_value_ub, w4, cp, False, False)
                            val = (w1[cp] * v1_ub + w2[cp] * v2_ub + w3[cp] * v3_ub + w4[cp] * v4_ub)
                            grad_weight_ub[cp, :] = top_grad_ub * val
                    grad_sample_x_loc_ub = top_grad_value_ub * grad_w_weight_ub * w
                    grad_sample_y_loc_ub = top_grad_value_ub * grad_h_weight_ub * h
                    x = np.sum(grad_sample_x_loc_ub, axis=-1)
                    y = np.sum(grad_sample_y_loc_ub, axis=-1)
                    weight_sum = np.sum(grad_weight_ub, axis=-1)
                    grad_attn_weight[data_weight_ptr + nl * num_point: data_weight_ptr + (nl + 1) * num_point] += weight_sum
                    grad_sampling_loc_out[data_loc_w_ptr + nl * 2 * num_point: data_loc_w_ptr + nl * 2 * num_point + num_point] += x
                    grad_sampling_loc_out[data_loc_w_ptr + nl * 2 * num_point + num_point: data_loc_w_ptr + nl * 2 * num_point + 2 * num_point] += y
    grad_sampling_loc_out = grad_sampling_loc_out.reshape((batch_size, num_query, num_heads, num_levels, 2, num_point)).transpose((0, 1, 2, 3, 5, 4))
    grad_value = grad_value.reshape((batch_size, spatial_size, num_heads, channels))
    grad_attn_weight = grad_attn_weight.reshape((batch_size, num_query, num_heads, num_levels, num_point))
    return grad_value, grad_sampling_loc_out, grad_attn_weight


class TestMultiScaleDeformableAttnGrad(TestCase):
    def gen_data(self, shape, dtype):
        bs, num_heads, embed_dims, num_levels, num_points, num_queries = shape
        cpu_shapes = torch.tensor([6, 4] * num_levels).reshape(num_levels, 2).int()
        cpu_shapes_numpy = cpu_shapes.numpy()
        num_keys = sum((H * W).item() for H, W in cpu_shapes).int()

        cpu_value = torch.rand(bs, num_keys, num_heads, embed_dims) * 0.01
        cpu_value_numpy = cpu_value.numpy()
        cpu_sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2)
        cpu_sampling_locations_numpy = cpu_sampling_locations.numpy()
        cpu_attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points) + 1e-5
        cpu_attention_weights_numpy = cpu_attention_weights.numpy()

        cpu_offset = torch.cat((cpu_shapes.new_zeros((1, )), cpu_shapes.prod(1).cumsum(0)[:-1]))
        cpu_offset_nunmpy = cpu_offset.numpy()
        grad_output = np.ones((bs, num_queries, num_heads, embed_dims)).astype(np.float)
        grad_value = np.ones_like(cpu_value_numpy)
        grad_sample_loc = np.ones_like(cpu_sampling_locations_numpy)
        grad_atten_weight = np.ones_like(cpu_attention_weights_numpy)
 
        npu_value = cpu_value.npu()
        npu_shapes = cpu_shapes.npu()
        npu_offset = cpu_offset.npu()
        npu_sampling_locations = cpu_sampling_locations.npu()
        npu_attention_weights = cpu_attention_weights.npu()
        npu_value.requires_grad = True
        npu_sampling_locations.requires_grad = True
        npu_attention_weights.requires_grad = True
        
        return [cpu_value_numpy, cpu_shapes_numpy, cpu_offset_nunmpy, cpu_sampling_locations_numpy, cpu_attention_weights_numpy, grad_output,
                grad_value, grad_sample_loc, grad_atten_weight], [npu_value, npu_shapes, npu_offset, npu_sampling_locations, npu_attention_weights]

    def cpu_to_exec(self, cpu_data):
        output1, output2, output3 = ms_scale_deform_att_grad(cpu_data[0], cpu_data[1], cpu_data[2], cpu_data[3], cpu_data[4], cpu_data[5], cpu_data[6], cpu_data[7], cpu_data[8])
        return output1, output2, output3

    def npu_to_exec(self, npu_data):
        output = ads.common.npu_multi_scale_deformable_attn_function(npu_data[0], npu_data[1], npu_data[2], npu_data[3], npu_data[4])
        a, b, c = output.shape
        grad_output = torch.ones((a,b,c)).float().npu()
        output.backward(grad_output)
        return npu_data[0].grad.cpu().numpy(), npu_data[3].grad.cpu().numpy(), npu_data[4].grad.cpu().numpy()

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `MultiScaleDeformableAttnFunction` is only supported on 910B, skip this ut!")
    def test_multi_scale_deformable_attn_function(self):
        dtype_list = [torch.float32]
        shape_list = [
            [[1, 8, 32, 1, 8, 968], [1, 8, 32, 1, 8, 308]]
        ]
        items = [
            [shape, dtype]
            for shape in shape_list
            for dtype in dtype_list
        ]
        for shape, dtype in items:
            cpu_x, npu_x = self.gen_data(shape, dtype)
            cpu_out1, cpu_out2, cpu_out2 = self.cpu_to_exec(cpu_x)
            npu_out1, npu_out2, npu_out3 = self.npu_to_exec(npu_x)
            self.assertRtolEqual(cpu_out1, npu_out1)
            self.assertRtolEqual(cpu_out2, npu_out2)
            self.assertRtolEqual(cpu_out3, npu_out3)


if __name__ == '__main__':
    run_tests()