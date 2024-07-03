import torch
import torch.nn as nn
from torch_npu.testing.testcase import TestCase, run_tests
import mx_driving.common


def gen_inputs(shape, dtype):
    x_data_cpu = torch.rand(shape, dtype=dtype)
    return x_data_cpu


def cpu_to_exec(x_data_cpu):
    f = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    cpu_output = f(x_data_cpu)
    return cpu_output


def npu_to_exec(x_data_cpu):
    npu_output = mx_driving.common.npu_max_pool2d(x_data_cpu.npu(), 3, 2, 1)
    return npu_output


class TestNpuMaxPool2d(TestCase):
    def test_npu_max_pool2d(self):
        dtype_list = [torch.float32]
        shape_list = [
            [18, 64, 464, 800],
            [6, 64, 464, 800],
            [1, 8, 3, 3]
        ]

        items = [
            [shape, dtype]
            for shape in shape_list
            for dtype in dtype_list
        ]

        for item in items:
            shape, dtype = item
            x_data_cpu = gen_inputs(shape, dtype)

            cpu_output = cpu_to_exec(x_data_cpu)
            npu_output = npu_to_exec(x_data_cpu)

            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
