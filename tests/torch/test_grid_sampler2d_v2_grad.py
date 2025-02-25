"""
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
import torch
import torch.nn.functional as F
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

from mx_driving import grid_sampler2d_v2


def gen_inputs(input_shape, grid_shape, dtype):
    input_tensor = torch.rand(input_shape, dtype=dtype, device='npu')
    rand_tensor = torch.rand(grid_shape, dtype=dtype, device='npu')
    # grid_tensor range: [-1, 1]
    grid_tensor = 2 * rand_tensor - 1
    inp_npu = torch.Tensor(input_tensor)
    grid_npu = torch.Tensor(grid_tensor)
    return input_tensor, grid_tensor, inp_npu, grid_npu


# pylint: disable=too-many-arguments,huawei-too-many-arguments
def gen_outputs(input_tensor, grid_tensor, inp_npu, grid_npu, mode, padding_mode, align_corners):
    input_tensor.requires_grad_()
    grid_tensor.requires_grad_()
    cpu_result = F.grid_sample(input_tensor, grid_tensor, mode, padding_mode, align_corners)
    cpu_result.backward(torch.ones_like(cpu_result))

    inp_npu.requires_grad_()
    grid_npu.requires_grad_()
    npu_result = grid_sampler2d_v2(inp_npu, grid_npu, mode, padding_mode, align_corners)
    npu_result.backward(torch.ones_like(npu_result))

    return input_tensor, grid_tensor, inp_npu, grid_npu


class TestGridSampler2dV2Grad(TestCase):
    seed = 1024
    torch.manual_seed(seed)

    def test_model_case(self, device="npu"):
        input_tensor, grid_tensor, inp_npu, grid_npu = gen_inputs([24, 4, 64, 176], [24, 5632, 176, 2], torch.float32)
        inp_cpu, grid_cpu, inp_npu, grid_npu = gen_outputs(input_tensor, grid_tensor, inp_npu, grid_npu, "bilinear", "zeros", True)
        self.assertRtolEqual(inp_cpu.grad.cpu().numpy(), inp_npu.grad.cpu().numpy())
        self.assertRtolEqual(grid_cpu.grad.cpu().numpy(), grid_npu.grad.cpu().numpy())

    def test_bilinear_zeros_false(self, device="npu"):
        input_tensor, grid_tensor, inp_npu, grid_npu = gen_inputs([24, 4, 64, 176], [24, 24, 176, 2], torch.float32)
        inp_cpu, grid_cpu, inp_npu, grid_npu = gen_outputs(input_tensor, grid_tensor, inp_npu, grid_npu, "bilinear", "zeros", False)
        self.assertRtolEqual(inp_cpu.grad.cpu().numpy(), inp_npu.grad.cpu().numpy())
        self.assertRtolEqual(grid_cpu.grad.cpu().numpy(), grid_npu.grad.cpu().numpy())

    def test_bilinear_zeros_true(self, device="npu"):
        input_tensor, grid_tensor, inp_npu, grid_npu = gen_inputs([24, 4, 64, 176], [24, 24, 176, 2], torch.float32)
        inp_cpu, grid_cpu, inp_npu, grid_npu = gen_outputs(input_tensor, grid_tensor, inp_npu, grid_npu, "bilinear", "zeros", True)
        self.assertRtolEqual(inp_cpu.grad.cpu().numpy(), inp_npu.grad.cpu().numpy())
        self.assertRtolEqual(grid_cpu.grad.cpu().numpy(), grid_npu.grad.cpu().numpy())

    def test_bilinear_border_false(self, device="npu"):
        input_tensor, grid_tensor, inp_npu, grid_npu = gen_inputs([24, 4, 64, 176], [24, 24, 176, 2], torch.float32)
        inp_cpu, grid_cpu, inp_npu, grid_npu = gen_outputs(input_tensor, grid_tensor, inp_npu, grid_npu, "bilinear", "border", False)
        self.assertRtolEqual(inp_cpu.grad.cpu().numpy(), inp_npu.grad.cpu().numpy())
        self.assertRtolEqual(grid_cpu.grad.cpu().numpy(), grid_npu.grad.cpu().numpy())

    def test_small_case(self, device="npu"):
        input_tensor, grid_tensor, inp_npu, grid_npu = gen_inputs([4, 4, 8, 8], [4, 16, 32, 2], torch.float32)
        inp_cpu, grid_cpu, inp_npu, grid_npu = gen_outputs(input_tensor, grid_tensor, inp_npu, grid_npu, "bilinear", "zeros", False)
        self.assertRtolEqual(inp_cpu.grad.cpu().numpy(), inp_npu.grad.cpu().numpy())
        self.assertRtolEqual(grid_cpu.grad.cpu().numpy(), grid_npu.grad.cpu().numpy())

    def test_channel_64(self, device="npu"):
        input_tensor, grid_tensor, inp_npu, grid_npu = gen_inputs([24, 64, 64, 176], [24, 24, 176, 2], torch.float32)
        inp_cpu, grid_cpu, inp_npu, grid_npu = gen_outputs(input_tensor, grid_tensor, inp_npu, grid_npu, "bilinear", "zeros", False)
        self.assertRtolEqual(inp_cpu.grad.cpu().numpy(), inp_npu.grad.cpu().numpy())
        self.assertRtolEqual(grid_cpu.grad.cpu().numpy(), grid_npu.grad.cpu().numpy())

    def test_channel_128(self, device="npu"):
        input_tensor, grid_tensor, inp_npu, grid_npu = gen_inputs([24, 64, 128, 176], [24, 24, 176, 2], torch.float32)
        inp_cpu, grid_cpu, inp_npu, grid_npu = gen_outputs(input_tensor, grid_tensor, inp_npu, grid_npu, "bilinear",
                                                           "zeros", False)
        self.assertRtolEqual(inp_cpu.grad.cpu().numpy(), inp_npu.grad.cpu().numpy())
        self.assertRtolEqual(grid_cpu.grad.cpu().numpy(), grid_npu.grad.cpu().numpy())

    def test_channel_20(self, device="npu"):
        input_tensor, grid_tensor, inp_npu, grid_npu = gen_inputs([24, 20, 20, 176], [24, 24, 176, 2], torch.float32)
        inp_cpu, grid_cpu, inp_npu, grid_npu = gen_outputs(input_tensor, grid_tensor, inp_npu, grid_npu, "bilinear",
                                                           "zeros", False)
        self.assertRtolEqual(inp_cpu.grad.cpu().numpy(), inp_npu.grad.cpu().numpy())
        self.assertRtolEqual(grid_cpu.grad.cpu().numpy(), grid_npu.grad.cpu().numpy())


if __name__ == "__main__":
    run_tests()
