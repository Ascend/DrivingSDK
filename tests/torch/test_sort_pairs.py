import unittest
from collections import namedtuple
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import mx_driving.common

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]

torch.manual_seed(3407)


def cpu_sort_pairs(keys_in, values_in, dim):
    keys_out, indices = torch.sort(keys_in, dim=dim, descending=False, stable=True)
    values_out = torch.gather(values_in, dim, indices)
    return keys_out, values_out


Inputs = namedtuple('Inputs', ['keys_in', 'values_in', 'dim'])


class TestSortPairs(TestCase):
    def setUp(self):
        self.dtype_list = [torch.uint8,
                           torch.int8,
                           torch.int16,
                           torch.int32,
                           torch.int64,
                           torch.bfloat16,
                           torch.float16,
                           torch.float32]
        self.shape_list = [
            [10, 200],
            [60, 300],
            [12, 400],
            [10, 2000],
            [20, 30, 100]
        ]
        self.items = [
            [shape, dtype]
            for shape in self.shape_list
            for dtype in self.dtype_list
        ]
        self.test_results = self.gen_results()
    
    def gen_results(self):
        if DEVICE_NAME != 'Ascend910B':
            self.skipTest("OP `SortPairs` is only supported on 910B, skipping test data generation!")
        test_results = []
        for shape, dtype in self.items:
            cpu_inputs, npu_inputs = self.gen_inputs(shape, dtype)
            cpu_results = self.cpu_to_exec(cpu_inputs)
            npu_results = self.npu_to_exec(npu_inputs)
            test_results.append((cpu_results, npu_results))
        return test_results
    
    def gen_inputs(self, shape, dtype):
        keys_in_cpu = torch.randint(-10000, 10000, shape).to(dtype)
        values_in_cpu = torch.randint(-10000, 10000, shape).to(dtype)
        
        keys_in_npu = keys_in_cpu.npu()
        values_in_npu = values_in_cpu.npu()

        dim = -1
        
        return Inputs(keys_in_cpu, values_in_cpu, dim), \
               Inputs(keys_in_npu, values_in_npu, dim)

    def cpu_to_exec(self, cpu_inputs):
        cpu_keys_in = cpu_inputs.keys_in
        cpu_values_in = cpu_inputs.values_in
        dim = cpu_inputs.dim
        cpu_keys_out, cpu_values_out = cpu_sort_pairs(cpu_keys_in, cpu_values_in, dim)
        return cpu_keys_out, cpu_values_out

    def npu_to_exec(self, npu_inputs):
        npu_keys_in = npu_inputs.keys_in
        npu_values_in = npu_inputs.values_in
        dim = npu_inputs.dim
        npu_keys_out, npu_values_out = mx_driving.common.sort_pairs(npu_keys_in, npu_values_in, dim)
        return npu_keys_out.cpu(), npu_values_out.cpu()

    def check_precision(self, actual, expected, rtol=1e-4, atol=1e-4, msg=None):
        if not torch.all(torch.isclose(actual, expected, rtol=rtol, atol=atol)):
            standardMsg = f'{actual} != {expected} within relative tolerance {rtol}'
            raise AssertionError(msg or standardMsg)

    def test_sort_pairs(self):
        for cpu_results, npu_results in self.test_results:
            self.check_precision(cpu_results[0], npu_results[0], 1e-20, 1e-20)
            self.check_precision(cpu_results[1], npu_results[1], 1e-20, 1e-20)
 

if __name__ == '__main__':
    run_tests()
