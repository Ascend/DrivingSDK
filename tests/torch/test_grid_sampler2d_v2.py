"""
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
import torch
import torch.nn.functional as F
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

from mx_driving import grid_sampler2d_v2


@golden_data_cache(__file__)
def gen_inputs(input_shape, grid_shape, dtype):
    input_tensor = torch.rand(input_shape, dtype=dtype)
    rand_tensor = torch.rand(grid_shape, dtype=dtype)
    # grid_tensor range: [-1, 1]
    grid_tensor = 2 * rand_tensor - 1
    return input_tensor, grid_tensor


@golden_data_cache(__file__)
def gen_outputs(input_tensor, grid_tensor, mode, padding_mode, align_corners):
    cpu_result = F.grid_sample(input_tensor, grid_tensor, mode, padding_mode, align_corners)
    npu_result = grid_sampler2d_v2(input_tensor.npu(), grid_tensor.npu(), mode, padding_mode, align_corners)
    return cpu_result, npu_result


class TestGridSampler2dV2(TestCase):
    seed = 1024
    torch.manual_seed(seed)

    def test_bilinear_zeros_false(self, device="npu"):
        input_tensor, grid_tensor = gen_inputs([24, 4, 64, 176], [24, 24, 176, 2], torch.float32)
        cpu_result, npu_result = gen_outputs(input_tensor, grid_tensor, "bilinear", "zeros", False)
        self.assertRtolEqual(cpu_result.cpu().detach().numpy(), npu_result.cpu().detach().numpy())

    def test_bilinear_zeros_true(self, device="npu"):
        input_tensor, grid_tensor = gen_inputs([24, 4, 64, 176], [24, 24, 176, 2], torch.float32)
        cpu_result, npu_result = gen_outputs(input_tensor, grid_tensor, "bilinear", "zeros", True)
        self.assertRtolEqual(cpu_result.cpu().detach().numpy(), npu_result.cpu().detach().numpy())

    def test_bilinear_border_false(self, device="npu"):
        input_tensor, grid_tensor = gen_inputs([24, 4, 64, 176], [24, 24, 176, 2], torch.float32)
        cpu_result, npu_result = gen_outputs(input_tensor, grid_tensor, "bilinear", "border", False)
        self.assertRtolEqual(cpu_result.cpu().detach().numpy(), npu_result.cpu().detach().numpy())

    def test_small_case(self, device="npu"):
        input_tensor, grid_tensor = gen_inputs([2, 4, 3, 4], [2, 2, 3, 2], torch.float32)
        cpu_result, npu_result = gen_outputs(input_tensor, grid_tensor, "bilinear", "zeros", False)
        self.assertRtolEqual(cpu_result.cpu().detach().numpy(), npu_result.cpu().detach().numpy())

    def test_channel_64(self, device="npu"):
        input_tensor, grid_tensor = gen_inputs([24, 64, 64, 176], [24, 24, 176, 2], torch.float32)
        cpu_result, npu_result = gen_outputs(input_tensor, grid_tensor, "bilinear", "zeros", False)
        self.assertRtolEqual(cpu_result.cpu().detach().numpy(), npu_result.cpu().detach().numpy())

    def test_channel_128(self, device="npu"):
        input_tensor, grid_tensor = gen_inputs([24, 128, 64, 176], [24, 24, 176, 2], torch.float32)
        cpu_result, npu_result = gen_outputs(input_tensor, grid_tensor, "bilinear", "zeros", False)
        self.assertRtolEqual(cpu_result.cpu().detach().numpy(), npu_result.cpu().detach().numpy())


if __name__ == "__main__":
    run_tests()
