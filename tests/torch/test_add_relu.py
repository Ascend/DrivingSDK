import unittest
import torch  
import numpy as np
import torch_npu
import torch.nn.functional as F

from torch_npu.testing.testcase import TestCase, run_tests
import mx_driving.common

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestPointsInBox(TestCase):  
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `AddRelu` is only supported on 910B, skip this ut!")
    def test_npu_add_relu_three_dim(self, device="npu"):
        x = np.random.uniform(1, 1, [1, 100, 3]).astype(np.float32)
        x = torch.from_numpy(x)
        y = np.random.uniform(2.0, 2.0, [1, 100, 3]).astype(np.float32)
        y = torch.from_numpy(y)
        cpu_result = F.relu(x + y)
        x = mx_driving.common.npu_add_relu(x.npu(), y.npu()).cpu().numpy()
        self.assertRtolEqual(x, cpu_result.numpy())
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `AddRelu` is only supported on 910B, skip this ut!")
    def test_npu_add_relu_large_number(self, device="npu"):
        x = np.random.uniform(1, 1, [18, 256, 232, 400]).astype(np.float32)
        x = torch.from_numpy(x)
        y = np.random.uniform(2.0, 2.0, [18, 256, 232, 400]).astype(np.float32)
        y = torch.from_numpy(y)
        cpu_result = F.relu(x + y)
        x = mx_driving.common.npu_add_relu(x.npu(), y.npu()).cpu().numpy()
        self.assertRtolEqual(x, cpu_result.numpy())
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `AddRelu` is only supported on 910B, skip this ut!")
    def test_npu_add_relu_fp16_large_number(self, device="npu"):
        x = np.random.uniform(1, 1, [18, 256, 232, 400]).astype(np.float16)
        x = torch.from_numpy(x)
        y = np.random.uniform(2.0, 2.0, [18, 256, 232, 400]).astype(np.float16)
        y = torch.from_numpy(y)
        cpu_result = F.relu(x.float() + y.float())
        x = mx_driving.common.npu_add_relu(x.npu(), y.npu()).cpu().numpy()
        self.assertRtolEqual(x, cpu_result.half().numpy())
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `AddRelu` is only supported on 910B, skip this ut!")
    def test_npu_add_relu_fp16_small_case(self, device="npu"):
        x = np.random.uniform(1, 1, [18]).astype(np.float16)
        x = torch.from_numpy(x)
        y = np.random.uniform(2.0, 2.0, [18]).astype(np.float16)
        y = torch.from_numpy(y)
        cpu_result = F.relu(x.float() + y.float())
        x = mx_driving.common.npu_add_relu(x.npu(), y.npu()).cpu().numpy()
        self.assertRtolEqual(x, cpu_result.half().numpy())


if __name__ == "__main__":
    run_tests()
