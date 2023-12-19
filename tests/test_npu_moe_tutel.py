import unittest
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import ads.common

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestMoeTutel(TestCase):
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def cpu_to_exec(self, x, gates, indices, locations, capacity, batch_size, sample_size, hidden, dtype):
        result = torch.zeros([batch_size, capacity, hidden]).to(dtype)
        for tensor_idx in range(batch_size):
            for i in range(sample_size):
                if locations[tensor_idx, i] < capacity and indices[tensor_idx, i] >= 0:
                    result[int(indices[tensor_idx, i]), int(locations[tensor_idx, i]), :] = gates[tensor_idx, i] * x[i,
                                                                                                                   :]
        return result

    def npu_to_exec(self, x, gates, indices, locations, capacity):
        out = ads.common.npu_moe_tutel(x, gates, indices, locations, capacity)
        return out.cpu()

    def gen_data(self, shape, dtype):
        cpu_input = torch.rand(shape, dtype=dtype)
        npu_input = cpu_input.npu()
        return cpu_input, npu_input
    
    def gen_data_gates(self, shape, dtype):
        cpu_input = torch.rand(shape).bool().to(dtype)
        npu_input = cpu_input.npu()
        return cpu_input, npu_input

    def gen_data_indices(self, shape):
        batch_size = shape[0]
        sample_size = shape[1]
        cpu_input = torch.zeros((1, sample_size)).int()
        indices = torch.ones((1, sample_size)).int()
        for i in range(batch_size - 1):
            cpu_input = torch.cat((cpu_input, torch.mul(indices, torch.tensor(i + 1, dtype=torch.int32))), 0)
        npu_input = cpu_input.npu()
        return cpu_input, npu_input

    def gen_data_locations(self, shape):
        batch_size = shape[0]
        sample_size = shape[1]
        cpu_input = torch.arange(0, sample_size).reshape(1, sample_size).int()
        cpu_input = cpu_input.repeat(batch_size, 1)
        npu_input = cpu_input.npu()
        return cpu_input, npu_input

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `MoeTutel` is only supported on 910B, skip this ut!")
    def test_moe_tutel(self):
        dtype_list = [torch.float16, torch.float32, torch.bfloat16]
        shape_list = [
            [[2, 5], [5, 16], 6],
            [[3, 6], [6, 16], 6],
            [[4, 7], [7, 32], 12],
            [[5, 8], [8, 32], 12],
            [[2, 16384], [16384, 32], 16384],
        ]
        items = [
            [shape, dtype]
            for shape in shape_list
            for dtype in dtype_list
        ]
        for shape, dtype in items:
            capacity = shape[2]
            batch_size = shape[0][0]
            sample_size = shape[0][1]
            hidden = shape[1][1]
            cpu_x, npu_x = self.gen_data(shape[1], dtype)
            cpu_gates, npu_gates = self.gen_data_gates(shape[0], dtype)
            cpu_indices, npu_indices = self.gen_data_indices(shape[0])
            cpu_locations, npu_locations = self.gen_data_locations(shape[0])
            cpu_out = self.cpu_to_exec(cpu_x, cpu_gates, cpu_indices, cpu_locations, capacity, batch_size, sample_size,
                                       hidden, dtype)
            npu_out = self.npu_to_exec(npu_x, npu_gates, npu_indices, npu_locations, capacity)
            if dtype == torch.bfloat16 or dtype == torch.float16:
                npu_out = npu_out.to(torch.float32)
                cpu_out = cpu_out.to(torch.float32)
            self.assertRtolEqual(npu_out.numpy(), cpu_out.numpy())


if __name__ == '__main__':
    run_tests()
