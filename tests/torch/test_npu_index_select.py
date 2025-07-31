import unittest
from copy import deepcopy

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving


@golden_data_cache(__file__)
def cpu_golden_outputs(x, dim, index):
    out = torch.index_select(x, dim, index).numpy()
    return out


class TestHypot(TestCase):
    def test_index_select(self):
        x = torch.randn(3, 4)
        index = torch.tensor([0, 2])
        gloden_output = cpu_golden_outputs(x, 0, index)
        npu_output = mx_driving.npu_index_select(x.npu(), 0, index.npu()).cpu()
        self.assertRtolEqual(npu_output.numpy(), gloden_output)

    def test_index_select_grad(self):
        x = torch.randn(3, 4)
        index = torch.tensor([0, 2])
        source = torch.randn([2, 4])
        x.requires_grad = True

        x_npu = deepcopy(x)
        index_npu = deepcopy(index)
        source_npu = deepcopy(source)

        torch.index_select(x, 0, index).backward(source)
        mx_driving.npu_index_select(x_npu.npu(), 0, index_npu.npu()).backward(source_npu.npu())

        self.assertRtolEqual(x.grad.numpy(), x_npu.grad.numpy())


if __name__ == "__main__":
    run_tests()
