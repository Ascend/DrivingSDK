"""
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
import torch
import torch.nn.functional as F
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

from mx_driving import grid_sampler3d_v1


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
    cann_result = F.grid_sample(input_tensor, grid_tensor, mode, padding_mode, align_corners)
    cann_result.backward(torch.ones_like(cann_result))

    inp_npu.requires_grad_()
    grid_npu.requires_grad_()
    npu_result = grid_sampler3d_v1(inp_npu, grid_npu, mode, padding_mode, align_corners)
    npu_result.backward(torch.ones_like(npu_result))

    return input_tensor, grid_tensor, inp_npu, grid_npu


class TestGridSampler3dGradV1(TestCase):
    seed = 1024
    torch.manual_seed(seed)

    def test_model_case(self, device="npu"):
        input_tensor, grid_tensor, inp_npu, grid_npu = gen_inputs([64, 64, 6, 64, 176], [64, 1460, 4, 1, 3], torch.float32)
        inp_cann, grid_cann, inp_npu, grid_npu = gen_outputs(input_tensor, grid_tensor, inp_npu, grid_npu, "bilinear", "zeros", True)
        self.assertRtolEqual(inp_cann.grad.cpu().numpy(), inp_npu.grad.cpu().numpy())
        self.assertRtolEqual(grid_cann.grad.cpu().numpy(), grid_npu.grad.cpu().numpy())

    def test_bilinear_zeros_true(self, device="npu"):
        input_tensor, grid_tensor, inp_npu, grid_npu = gen_inputs([24, 64, 6, 64, 176], [24, 24, 4, 1, 3], torch.float32)
        inp_cann, grid_cann, inp_npu, grid_npu = gen_outputs(input_tensor, grid_tensor, inp_npu, grid_npu, "bilinear", "zeros", True)
        self.assertRtolEqual(inp_cann.grad.cpu().numpy(), inp_npu.grad.cpu().numpy())
        self.assertRtolEqual(grid_cann.grad.cpu().numpy(), grid_npu.grad.cpu().numpy())

    def test_small_case(self, device="npu"):
        input_tensor, grid_tensor, inp_npu, grid_npu = gen_inputs([2, 6, 4, 4, 4], [2, 1, 4, 2, 3], torch.float32)
        inp_cann, grid_cann, inp_npu, grid_npu = gen_outputs(input_tensor, grid_tensor, inp_npu, grid_npu, "bilinear", "zeros", True)
        self.assertRtolEqual(inp_cann.grad.cpu().numpy(), inp_npu.grad.cpu().numpy())
        self.assertRtolEqual(grid_cann.grad.cpu().numpy(), grid_npu.grad.cpu().numpy())

if __name__ == "__main__":
    run_tests()