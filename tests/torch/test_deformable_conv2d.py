import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from mmcv.ops import deform_conv2d as mmcv_deform_conv2d
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving
from mx_driving import deform_conv2d


class TestDeformableConv2d(TestCase):

    @golden_data_cache(__file__)
    def create_single_cpu_tensor(self, item, minvalue, maxvalue):
        dtype = item[0]
        format1 = item[1]
        shape = item[2]
        input1 = np.random.uniform(minvalue, maxvalue, shape).astype(dtype)
        return torch.from_numpy(input1)

    @golden_data_cache(__file__)
    def get_cpu_golden(self, x, offset, weight, groups):
        x_npu = x.clone()
        offset_npu = offset.clone()
        weight_npu = weight.clone()
        x_npu.grad, offset_npu.grad, weight_npu.grad = None, None, None

        x_npu.requires_grad = True
        offset_npu.requires_grad = True
        weight_npu.requires_grad = True

        out = mmcv_deform_conv2d(x_npu, offset_npu, weight_npu, 1, 1, 1, groups)
        out.backward(torch.ones_like(out), retain_graph=True)

        return out.detach(), x_npu.grad.detach(), offset_npu.grad.detach(), weight_npu.grad.detach()

    def get_npu_output(self, x, offset, weight, groups):
        x_npu = x.clone().npu()
        offset_npu = offset.clone().npu()
        weight_npu = weight.clone().npu()
        x_npu.grad, offset_npu.grad, weight_npu.grad = None, None, None

        x_npu.requires_grad = True
        offset_npu.requires_grad = True
        weight_npu.requires_grad = True

        out = deform_conv2d(x_npu, offset_npu, weight_npu, 1, 1, 1, groups)
        out.backward(torch.ones_like(out), retain_graph=True)

        return (
            out.detach().cpu(),
            x_npu.grad.detach().cpu(),
            offset_npu.grad.detach().cpu(),
            weight_npu.grad.detach().cpu(),
        )

    def test_deformable_conv2d_single_group(self):
        N, cIn, cOut, K, hIn, wIn, hOut, wOut, groups = 18, 512, 512, 3, 29, 50, 29, 50, 1

        cpu_x = self.create_single_cpu_tensor([np.float32, 0, (N, cIn, hIn, wIn)], -5, 5)
        cpu_w = self.create_single_cpu_tensor([np.float32, 0, (cOut, cIn // groups, K, K)], -5, 5) * 0.01
        cpu_o = self.create_single_cpu_tensor([np.float32, 0, (N, 2 * K * K, hOut, wOut)], -5, 5)
        out_cpu, x_grad_cpu, offset_grad_cpu, weight_grad_cpu = self.get_cpu_golden(cpu_x, cpu_o, cpu_w, groups)
        out_npu, x_grad_npu, offset_grad_npu, weight_grad_npu = self.get_npu_output(cpu_x, cpu_o, cpu_w, groups)

        self.assertRtolEqual(out_npu, out_cpu)
        self.assertRtolEqual(x_grad_npu, x_grad_cpu)
        self.assertRtolEqual(offset_grad_npu, offset_grad_cpu, 1e-3, 1e-3)
        self.assertRtolEqual(weight_grad_npu, weight_grad_cpu, 1e-2, 1e-2)

    def test_deformable_conv2d_multi_group(self):
        N, cIn, cOut, K, hIn, wIn, hOut, wOut, groups = 18, 512, 512, 3, 29, 50, 29, 50, 8

        cpu_x = self.create_single_cpu_tensor([np.float32, 0, (N, cIn, hIn, wIn)], -5, 5)
        cpu_w = self.create_single_cpu_tensor([np.float32, 0, (cOut, cIn // groups, K, K)], -5, 5) * 0.01
        cpu_o = self.create_single_cpu_tensor([np.float32, 0, (N, 2 * K * K, hOut, wOut)], -5, 5)
        out_cpu, x_grad_cpu, offset_grad_cpu, weight_grad_cpu = self.get_cpu_golden(cpu_x, cpu_o, cpu_w, groups)
        out_npu, x_grad_npu, offset_grad_npu, weight_grad_npu = self.get_npu_output(cpu_x, cpu_o, cpu_w, groups)

        self.assertRtolEqual(out_npu, out_cpu)
        self.assertRtolEqual(x_grad_npu, x_grad_cpu)
        self.assertRtolEqual(offset_grad_npu, offset_grad_cpu, 1e-3, 1e-3)
        self.assertRtolEqual(weight_grad_npu, weight_grad_cpu, 1e-2, 1e-2)


if __name__ == "__main__":
    run_tests()
