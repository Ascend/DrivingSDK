import unittest

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving.fused


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


@golden_data_cache(__file__)
def gen_inputs(shape, dtype):
    x = np.random.uniform(1, 1, shape).astype(dtype)
    x = torch.from_numpy(x)
    y = np.random.uniform(1, 1, shape).astype(dtype)
    y = torch.from_numpy(y)
    return x, y


@golden_data_cache(__file__)
def gen_cpu_outputs(x, y):
    cpu_result = F.relu(x.float() + y.float())
    return cpu_result


class TestAddRelu(TestCase):  
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `AddRelu` is only supported on 910B, skip this ut!")
    def test_npu_add_relu_three_dim(self, device="npu"):
        x, y = gen_inputs([1, 100, 3], np.float32)
        cpu_result = gen_cpu_outputs(x, y)
        result = mx_driving.fused.npu_add_relu(x.npu(), y.npu()).cpu().numpy()
        self.assertRtolEqual(result, cpu_result.numpy())
        result = mx_driving.npu_add_relu(x.npu(), y.npu()).cpu().numpy()
        self.assertRtolEqual(result, cpu_result.numpy())
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `AddRelu` is only supported on 910B, skip this ut!")
    def test_npu_add_relu_large_number(self, device="npu"):
        x, y = gen_inputs([18, 256, 232, 100], np.float32)
        cpu_result = gen_cpu_outputs(x, y)
        result = mx_driving.fused.npu_add_relu(x.npu(), y.npu()).cpu().numpy()
        self.assertRtolEqual(result, cpu_result.numpy())
        result = mx_driving.npu_add_relu(x.npu(), y.npu()).cpu().numpy()
        self.assertRtolEqual(result, cpu_result.numpy())
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `AddRelu` is only supported on 910B, skip this ut!")
    def test_npu_add_relu_fp16_large_number(self, device="npu"):
        x, y = gen_inputs([18, 256, 232, 100], np.float16)
        cpu_result = gen_cpu_outputs(x, y)
        result = mx_driving.fused.npu_add_relu(x.npu(), y.npu()).cpu().numpy()
        self.assertRtolEqual(result, cpu_result.half().numpy())
        result = mx_driving.npu_add_relu(x.npu(), y.npu()).cpu().numpy()
        self.assertRtolEqual(result, cpu_result.half().numpy())
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `AddRelu` is only supported on 910B, skip this ut!")
    def test_npu_add_relu_fp16_small_case(self, device="npu"):
        x, y = gen_inputs([18], np.float16)
        cpu_result = gen_cpu_outputs(x, y)
        result = mx_driving.fused.npu_add_relu(x.npu(), y.npu()).cpu().numpy()
        self.assertRtolEqual(result, cpu_result.half().numpy())
        result = mx_driving.npu_add_relu(x.npu(), y.npu()).cpu().numpy()
        self.assertRtolEqual(result, cpu_result.half().numpy())


if __name__ == "__main__":
    run_tests()
