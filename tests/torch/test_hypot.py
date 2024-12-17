import unittest
from copy import deepcopy

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


@golden_data_cache(__file__)
def cpu_gen_inputs(data_range_x, data_range_y, x_shape, y_shape):
    x = np.random.uniform(data_range_x[0], data_range_x[1], x_shape).astype(np.float32)
    x = torch.from_numpy(x)
    y = np.random.uniform(data_range_y[0], data_range_y[1], y_shape).astype(np.float32)
    y = torch.from_numpy(y)

    return x, y


@golden_data_cache(__file__)
def cpu_gen_outputs(x, y):
    z = torch.hypot(x, y).numpy()
    return z


class TestHypot(TestCase):
    def test_hypot_one_dim(self, device="npu"):
        x, y = cpu_gen_inputs([3, 3], [4, 4], [1], [1])
        z = cpu_gen_outputs(x, y)
        npu_result = mx_driving.hypot(x.npu(), y.npu()).cpu()
        self.assertRtolEqual(npu_result.numpy(), z)

    def test_hypot_one_dim_broadcast(self, device="npu"):
        x, y = cpu_gen_inputs([3, 3], [4, 4], [1], [10])
        z = cpu_gen_outputs(x, y)
        npu_result = mx_driving.hypot(x.npu(), y.npu()).cpu()
        self.assertRtolEqual(npu_result.numpy(), z)

    def test_hypot_three_dim(self, device="npu"):
        x, y = cpu_gen_inputs([3, 3], [4, 4], [35, 50, 80], [35, 50, 80])
        z = cpu_gen_outputs(x, y)
        npu_result = mx_driving.hypot(x.npu(), y.npu()).cpu()
        self.assertRtolEqual(npu_result.numpy(), z)

    def test_hypot_random_three_dim(self, device="npu"):
        x, y = cpu_gen_inputs([1, 3], [1, 4], [35, 50, 80], [35, 50, 80])
        z = cpu_gen_outputs(x, y)
        npu_result = mx_driving.hypot(x.npu(), y.npu()).cpu()
        self.assertRtolEqual(npu_result.numpy(), z)

    def test_hypot_random_three_dim_broadcast_x(self, device="npu"):
        x, y = cpu_gen_inputs([1, 3], [1, 4], [35, 1, 80], [35, 50, 80])
        z = cpu_gen_outputs(x, y)
        npu_result = mx_driving.hypot(x.npu(), y.npu()).cpu()
        self.assertRtolEqual(npu_result.numpy(), z)

    def test_hypot_random_three_dim_broadcast_y(self, device="npu"):
        x, y = cpu_gen_inputs([1, 3], [1, 4], [35, 50, 80], [35, 1, 80])
        z = cpu_gen_outputs(x, y)
        npu_result = mx_driving.hypot(x.npu(), y.npu()).cpu()
        self.assertRtolEqual(npu_result.numpy(), z)

    def test_hypot_large_random_dim_broadcast(self, device="npu"):
        x, y = cpu_gen_inputs([1, 3], [1, 4], [35, 50, 80, 1, 3], [35, 1, 80, 171, 3])
        z = cpu_gen_outputs(x, y)
        npu_result = mx_driving.hypot(x.npu(), y.npu()).cpu()
        self.assertRtolEqual(npu_result.numpy(), z)

    def test_hypot_grad_base(self, device="npu"):
        x, y = cpu_gen_inputs([3, 3], [4, 4], [35, 50], [35, 50])
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
        x, y = cpu_gen_inputs([0, 0], [0, 0], [35, 50], [35, 50])
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
        x, y = cpu_gen_inputs([-3, 3], [-4, 4], [35, 50, 80, 1, 3], [35, 1, 80, 171, 3])
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
