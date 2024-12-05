from copy import deepcopy
import unittest
import torch
import torch_npu
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests
import mx_driving

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestHypot(TestCase):
    def test_hypot_one_dim(self, device="npu"):
        x = np.random.uniform(3, 3, [1]).astype(np.float32)
        x = torch.from_numpy(x)
        y = np.random.uniform(4, 4, [1]).astype(np.float32)
        y = torch.from_numpy(y)
        z = np.random.uniform(5, 5, [1]).astype(np.float32)
        npu_result = mx_driving.hypot(x.npu(), y.npu()).cpu()
        self.assertRtolEqual(npu_result.numpy(), z)

    def test_hypot_one_dim_broadcast(self, device="npu"):
        x = np.random.uniform(3, 3, [1]).astype(np.float32)
        x = torch.from_numpy(x)
        y = np.random.uniform(4, 4, [10]).astype(np.float32)
        y = torch.from_numpy(y)
        z = np.random.uniform(5, 5, [10]).astype(np.float32)
        npu_result = mx_driving.hypot(x.npu(), y.npu()).cpu()
        self.assertRtolEqual(npu_result.numpy(), z)

    def test_hypot_three_dim(self, device="npu"):
        x = np.random.uniform(3, 3, [35, 50, 80]).astype(np.float32)
        x = torch.from_numpy(x)
        y = np.random.uniform(4, 4, [35, 50, 80]).astype(np.float32)
        y = torch.from_numpy(y)
        z = np.random.uniform(5, 5, [35, 50, 80]).astype(np.float32)
        npu_result = mx_driving.hypot(x.npu(), y.npu()).cpu()
        self.assertRtolEqual(npu_result.numpy(), z)

    def test_hypot_random_three_dim(self, device="npu"):
        x = np.random.uniform(1, 3, [35, 50, 80]).astype(np.float32)
        x = torch.from_numpy(x)
        y = np.random.uniform(1, 4, [35, 50, 80]).astype(np.float32)
        y = torch.from_numpy(y)
        z = torch.hypot(x, y).numpy()
        npu_result = mx_driving.hypot(x.npu(), y.npu()).cpu()
        self.assertRtolEqual(npu_result.numpy(), z)

    def test_hypot_random_three_dim_broadcast_x(self, device="npu"):
        x = np.random.uniform(1, 3, [35, 1, 80]).astype(np.float32)
        x = torch.from_numpy(x)
        y = np.random.uniform(1, 4, [35, 50, 80]).astype(np.float32)
        y = torch.from_numpy(y)
        z = torch.hypot(x, y).numpy()
        npu_result = mx_driving.hypot(x.npu(), y.npu()).cpu()
        self.assertRtolEqual(npu_result.numpy(), z)

    def test_hypot_random_three_dim_broadcast_y(self, device="npu"):
        x = np.random.uniform(1, 3, [35, 50, 80]).astype(np.float32)
        x = torch.from_numpy(x)
        y = np.random.uniform(1, 4, [35, 1, 80]).astype(np.float32)
        y = torch.from_numpy(y)
        z = torch.hypot(x, y).numpy()
        npu_result = mx_driving.hypot(x.npu(), y.npu()).cpu()
        self.assertRtolEqual(npu_result.numpy(), z)

    def test_hypot_large_random_dim_broadcast(self, device="npu"):
        x = np.random.uniform(1, 3, [35, 50, 80, 1, 3]).astype(np.float32)
        x = torch.from_numpy(x)
        y = np.random.uniform(1, 4, [35, 1, 80, 171, 3]).astype(np.float32)
        y = torch.from_numpy(y)
        z = torch.hypot(x, y).numpy()
        npu_result = mx_driving.hypot(x.npu(), y.npu()).cpu()
        self.assertRtolEqual(npu_result.numpy(), z)

    def test_hypot_grad_base(self, device="npu"):
        x = np.random.uniform(3, 3, [35, 50]).astype(np.float32)
        x = torch.from_numpy(x)
        y = np.random.uniform(4, 4, [35, 50]).astype(np.float32)
        y = torch.from_numpy(y)
        z_grad = torch.randn([35, 50])
        x.requires_grad = True
        y.requires_grad = True
        x_npu = deepcopy(x)
        y_npu = deepcopy(y)

        torch.hypot(x, y).backward(z_grad)
        mx_driving.hypot(x_npu.npu(), y_npu.npu()).backward(z_grad.npu())

        self.assertRtolEqual(x.grad.numpy(), x_npu.grad.numpy())
        self.assertRtolEqual(y.grad.numpy(), y_npu.grad.numpy())

    def test_hypot_grad_zero(self, device="npu"):
        x = np.random.uniform(0, 0, [35, 50]).astype(np.float32)
        x = torch.from_numpy(x)
        y = np.random.uniform(0, 0, [35, 50]).astype(np.float32)
        y = torch.from_numpy(y)
        z_grad = torch.randn([35, 50])
        x.requires_grad = True
        y.requires_grad = True
        x_npu = deepcopy(x)
        y_npu = deepcopy(y)

        torch.hypot(x, y).backward(z_grad)
        mx_driving.hypot(x_npu.npu(), y_npu.npu()).backward(z_grad.npu())

        self.assertRtolEqual(x.grad.numpy(), x_npu.grad.numpy())
        self.assertRtolEqual(y.grad.numpy(), y_npu.grad.numpy())

    def test_hypot_grad_large_random_dim_broadcast(self, device="npu"):
        x = np.random.uniform(-3, 3, [35, 50, 80, 1, 3]).astype(np.float32)
        x = torch.from_numpy(x)
        y = np.random.uniform(-4, 4, [35, 1, 80, 171, 3]).astype(np.float32)
        y = torch.from_numpy(y)
        z_grad = torch.randn([35, 50, 80, 171, 3])
        x.requires_grad = True
        y.requires_grad = True
        x_npu = deepcopy(x)
        y_npu = deepcopy(y)

        torch.hypot(x, y).backward(z_grad)
        mx_driving.hypot(x_npu.npu(), y_npu.npu()).backward(z_grad.npu())

        self.assertRtolEqual(x.grad.numpy(), x_npu.grad.numpy())
        self.assertRtolEqual(y.grad.numpy(), y_npu.grad.numpy())

if __name__ == "__main__":
    run_tests()
