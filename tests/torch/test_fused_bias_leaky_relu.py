import unittest
import torch  
import numpy as np
import torch_npu
import torch.nn.functional as F

from torch_npu.testing.testcase import TestCase, run_tests
import mx_driving.common

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]

negative_slop = -0.1
scale = 0.25


class TestFusedBiasLeakyRelu(TestCase):
    seed = 1024
    np.random.seed(seed)

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `FusedBiasLeakyRelu` is only supported on 910B, skip this ut!")
    def test_npu_fused_bias_leaky_relu_three_dim(self, device="npu"):
        x = np.random.uniform(1, 1, [1, 100, 3]).astype(np.float32)
        x = torch.from_numpy(x)
        bias = np.random.uniform(2.0, 2.0, [1, 100, 3]).astype(np.float32)
        bias = torch.from_numpy(bias)
        
        cpu_result = F.leaky_relu(x + bias, negative_slop)
        cpu_result = cpu_result * scale

        npu_result = mx_driving.common.npu_fused_bias_leaky_relu(x.npu(), bias.npu(), negative_slop, scale).cpu().numpy()
        self.assertRtolEqual(npu_result, cpu_result.numpy())
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `FusedBiasLeakyRelu` is only supported on 910B, skip this ut!")
    def test_npu_fused_bias_leaky_relu_large_number(self, device="npu"):
        x = np.random.uniform(1, 1, [18, 256, 232, 400]).astype(np.float32)
        x = torch.from_numpy(x)
        bias = np.random.uniform(2.0, 2.0, [18, 256, 232, 400]).astype(np.float32)
        bias = torch.from_numpy(bias)

        cpu_result = F.leaky_relu(x + bias, negative_slop)
        cpu_result = cpu_result * scale

        npu_result = mx_driving.common.npu_fused_bias_leaky_relu(x.npu(), bias.npu(), negative_slop, scale).cpu().numpy()
        self.assertRtolEqual(npu_result, cpu_result.numpy())
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `FusedBiasLeakyRelu` is only supported on 910B, skip this ut!")
    def test_npu_fused_bias_leaky_relu_fp16_large_number(self, device="npu"):
        x = np.random.uniform(1, 1, [18, 256, 232, 400]).astype(np.float16)
        x = torch.from_numpy(x)
        bias = np.random.uniform(2.0, 2.0, [18, 256, 232, 400]).astype(np.float16)
        bias = torch.from_numpy(bias)

        cpu_result = F.leaky_relu(x.float() + bias.float(), negative_slop)
        cpu_result = cpu_result * scale

        npu_result = mx_driving.common.npu_fused_bias_leaky_relu(x.npu(), bias.npu(), negative_slop, scale).cpu().numpy()
        self.assertRtolEqual(npu_result, cpu_result.half().numpy())
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `FusedBiasLeakyRelu` is only supported on 910B, skip this ut!")
    def test_npu_fused_bias_leaky_relu_fp16_small_case(self, device="npu"):
        x = np.random.uniform(1, 1, [18]).astype(np.float16)
        x = torch.from_numpy(x)
        bias = np.random.uniform(2.0, 2.0, [18]).astype(np.float16)
        bias = torch.from_numpy(bias)

        cpu_result = F.leaky_relu(x.float() + bias.float(), negative_slop)
        cpu_result = cpu_result * scale

        npu_result = mx_driving.common.npu_fused_bias_leaky_relu(x.npu(), bias.npu(), negative_slop, scale).cpu().numpy()
        self.assertRtolEqual(npu_result, cpu_result.half().numpy())


if __name__ == "__main__":
    run_tests()