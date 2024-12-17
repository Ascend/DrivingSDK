import unittest

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving.fused


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]

negative_slop = -0.1
scale = 0.25


@golden_data_cache(__file__)
def cpu_gen_inputs(shape, bias_dim, feature_dtype, bias_dtype):
    x = np.random.uniform(1, 1, shape).astype(feature_dtype)
    x = torch.from_numpy(x)
    bias = np.random.uniform(-2.0, 2.0, bias_dim).astype(bias_dtype)
    bias = torch.from_numpy(bias)
    bias_cpu = bias.reshape([-1 if i == 1 else 1 for i in range(x.ndim)])
    bias_cpu = bias_cpu.repeat([1 if i == 1 else x.size(i) for i in range(x.ndim)])

    return x, bias, bias_cpu


@golden_data_cache(__file__)
def cpu_gen_outputs(x, bias_cpu):
    print(x.shape, bias_cpu.shape)
    cpu_result = F.leaky_relu(x.float() + bias_cpu.float(), negative_slop)
    cpu_result = cpu_result * scale

    return cpu_result


class TestFusedBiasLeakyRelu(TestCase):
    seed = 1024
    np.random.seed(seed)

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `FusedBiasLeakyRelu` is only supported on 910B, skip this ut!")
    def test_npu_fused_bias_leaky_relu_three_dim(self, device="npu"):
        N, H, W = [1, 100, 3]
        x, bias, bias_cpu = cpu_gen_inputs([1, 100, 3], H, np.float32, np.float32)
        
        cpu_result = cpu_gen_outputs(x, bias_cpu)

        npu_result = mx_driving.fused.npu_fused_bias_leaky_relu(x.npu(), bias.npu(), negative_slop, scale).cpu().numpy()
        self.assertRtolEqual(npu_result, cpu_result.numpy())
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `FusedBiasLeakyRelu` is only supported on 910B, skip this ut!")
    def test_npu_fused_bias_leaky_relu_large_number(self, device="npu"):
        B, N, H, W = [18, 256, 232, 100]
        x, bias, bias_cpu = cpu_gen_inputs([18, 256, 232, 100], N, np.float32, np.float32)

        cpu_result = cpu_gen_outputs(x, bias_cpu)

        npu_result = mx_driving.fused.npu_fused_bias_leaky_relu(x.npu(), bias.npu(), negative_slop, scale).cpu().numpy()
        self.assertRtolEqual(npu_result, cpu_result.numpy())
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `FusedBiasLeakyRelu` is only supported on 910B, skip this ut!")
    def test_npu_fused_bias_leaky_relu_fp16_large_number(self, device="npu"):
        B, N, H, W = [18, 256, 232, 100]
        x, bias, bias_cpu = cpu_gen_inputs([18, 256, 232, 100], N, np.float16, np.float32)

        cpu_result = cpu_gen_outputs(x, bias_cpu)

        npu_result = mx_driving.fused.npu_fused_bias_leaky_relu(x.npu(), bias.npu(), negative_slop, scale).cpu().numpy()
        self.assertRtolEqual(npu_result, cpu_result.half().numpy())
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `FusedBiasLeakyRelu` is only supported on 910B, skip this ut!")
    def test_npu_fused_bias_leaky_relu_fp16_small_case(self, device="npu"):
        N, H, W = [18, 200, 6]
        x, bias, bias_cpu = cpu_gen_inputs([18, 200, 6], H, np.float16, np.float32)

        cpu_result = cpu_gen_outputs(x, bias_cpu)

        npu_result = mx_driving.fused.npu_fused_bias_leaky_relu(x.npu(), bias.npu(), negative_slop, scale).cpu().numpy()
        self.assertRtolEqual(npu_result, cpu_result.half().numpy())


if __name__ == "__main__":
    run_tests()