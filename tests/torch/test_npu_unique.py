"""
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
import random
import os
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests
from mx_driving import npu_unique


def gen_inputs(input_shape, dtype):
    input_tensor = torch.randint(-256, 256, input_shape, dtype=dtype)
    return input_tensor


def gen_cpu_outputs(input_tensor):
    cpu_result = torch.unique(input_tensor)
    return cpu_result


def gen_npu_outputs(input_tensor):
    npu_result = npu_unique(input_tensor.npu())
    return npu_result.cpu()


class TestNpuUnique(TestCase):
    def test_bfloat16(self, device='npu'):
        input_tensor = gen_inputs([random.randint(1, 10000000)], torch.bfloat16)
        cpu_result = gen_cpu_outputs(input_tensor)
        npu_result = gen_npu_outputs(input_tensor)
        self.assertRtolEqual(cpu_result.cpu().detach().to(torch.float32).numpy(),
                             npu_result.cpu().detach().to(torch.float32).numpy())

    def test_float16(self, device='npu'):
        input_tensor = gen_inputs([random.randint(1, 10000000)], torch.float16)
        cpu_result = gen_cpu_outputs(input_tensor)
        npu_result = gen_npu_outputs(input_tensor)
        self.assertRtolEqual(cpu_result.cpu().detach().numpy(), npu_result.cpu().detach().numpy())

    def test_float32(self, device='npu'):
        input_tensor = gen_inputs([random.randint(1, 10000000)], torch.float32)
        cpu_result = gen_cpu_outputs(input_tensor)
        npu_result = gen_npu_outputs(input_tensor)
        self.assertRtolEqual(cpu_result.cpu().detach().numpy(), npu_result.cpu().detach().numpy())

    def test_int16(self, device='npu'):
        input_tensor = gen_inputs([random.randint(1, 10000000)], torch.int16)
        cpu_result = gen_cpu_outputs(input_tensor)
        npu_result = gen_npu_outputs(input_tensor)
        self.assertRtolEqual(cpu_result.cpu().detach().numpy(), npu_result.cpu().detach().numpy())

    def test_int32(self, device='npu'):
        input_tensor = gen_inputs([random.randint(1, 10000000)], torch.int32)
        cpu_result = gen_cpu_outputs(input_tensor)
        npu_result = gen_npu_outputs(input_tensor)
        self.assertRtolEqual(cpu_result.cpu().detach().numpy(), npu_result.cpu().detach().numpy())

    def test_int64(self, device='npu'):
        input_tensor = gen_inputs([random.randint(1, 10000000)], torch.int64)
        cpu_result = gen_cpu_outputs(input_tensor)
        npu_result = gen_npu_outputs(input_tensor)
        self.assertRtolEqual(cpu_result.cpu().detach().numpy(), npu_result.cpu().detach().numpy())


if __name__ == "__main__":
    run_tests()