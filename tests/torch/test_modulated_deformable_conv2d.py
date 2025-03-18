import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from mmcv.ops import modulated_deform_conv2d as mmcv_modulated_deform_conv2d
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving
from mx_driving import modulated_deform_conv2d


class TestModulatedDeformableConv2d(TestCase):

    @golden_data_cache(__file__)
    def create_single_cpu_tensor(self, item, minvalue, maxvalue):
        dtype = item[0]
        format1 = item[1]
        shape = item[2]
        input1 = np.random.uniform(minvalue, maxvalue, shape).astype(dtype)
        return torch.from_numpy(input1)

    @golden_data_cache(__file__)
    def get_fwd_golden(self, x, offset, mask, weight, groups):
        return mmcv_modulated_deform_conv2d(x, offset, mask, weight, None, 1, 1, 1, groups)

    def test_modulated_deformable_conv2d_single_group(self):
        N, cIn, cOut, K, hIn, wIn, hOut, wOut, groups = 18, 512, 512, 3, 29, 50, 29, 50, 1

        cpu_x = self.create_single_cpu_tensor([np.float32, 0, (N, cIn, hIn, wIn)], -5, 5)
        cpu_w = self.create_single_cpu_tensor([np.float32, 0, (cOut, cIn // groups, K, K)], -5, 5) * 0.01
        cpu_o = self.create_single_cpu_tensor([np.float32, 0, (N, 2 * K * K, hOut, wOut)], -5, 5)
        cpu_m = self.create_single_cpu_tensor([np.float32, 0, (N, K * K, hOut, wOut)], -5, 5)
        cpu_output = self.get_fwd_golden(cpu_x, cpu_o, cpu_m, cpu_w, groups)

        output = modulated_deform_conv2d(cpu_x.npu(), cpu_o.npu(), cpu_m.npu(), cpu_w.npu(), None, 1, 1, 1, groups)
        self.assertRtolEqual(output, cpu_output)

    def test_modulated_deformable_conv2d_multi_grad(self):
        N, cIn, cOut, K, hIn, wIn, hOut, wOut, groups = 18, 512, 512, 3, 29, 50, 29, 50, 8

        cpu_x = self.create_single_cpu_tensor([np.float32, 0, (N, cIn, hIn, wIn)], -5, 5)
        cpu_w = self.create_single_cpu_tensor([np.float32, 0, (cOut, cIn // groups, K, K)], -5, 5) * 0.01
        cpu_o = self.create_single_cpu_tensor([np.float32, 0, (N, 2 * K * K, hOut, wOut)], -5, 5)
        cpu_m = self.create_single_cpu_tensor([np.float32, 0, (N, K * K, hOut, wOut)], -5, 5)
        cpu_output = self.get_fwd_golden(cpu_x, cpu_o, cpu_m, cpu_w, groups)

        output = modulated_deform_conv2d(cpu_x.npu(), cpu_o.npu(), cpu_m.npu(), cpu_w.npu(), None, 1, 1, 1, groups)
        self.assertRtolEqual(output, cpu_output)


if __name__ == "__main__":
    run_tests()
