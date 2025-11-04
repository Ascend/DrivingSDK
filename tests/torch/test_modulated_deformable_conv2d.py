import os
import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from mmcv.ops import modulated_deform_conv2d as mmcv_modulated_deform_conv2d
from torch_npu.testing.testcase import TestCase, run_tests

from cv_fused_double_benchmark_compare import CvFusedDoubleBenchmarkAccuracyCompare
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

    def test_k_not_equal_three_raises_exception(self):
        # k not equal to 3
        N, cIn, cOut, K, hIn, wIn, hOut, wOut, groups = 18, 512, 512, 5, 29, 50, 29, 50, 1

        x = self.create_single_cpu_tensor([np.float32, 0, (N, cIn, hIn, wIn)], -5, 5)
        offset = self.create_single_cpu_tensor([np.float32, 0, (N, 2 * K * K, hOut, wOut)], -2, 2)
        mask = self.create_single_cpu_tensor([np.float32, 0, (N, K * K, hOut, wOut)], -5, 5)
        weight = self.create_single_cpu_tensor([np.float32, 0, (cOut, cIn // groups, K, K)], -5, 5) * 0.01
        npu_x, npu_offset, npu_mask, npu_weight = x.npu(), offset.npu(), mask.npu(), weight.npu()

        try:
            npu_out = modulated_deform_conv2d(npu_x, npu_offset, npu_mask, npu_weight, None, 1, 1, 1, groups).cpu()
        except ValueError as e:
            assert str(e) == "Kernel size only support 3"

    def test_channel_not_aligned_raises_exception(self):
        # cIn, cOut not 64-Aligned
        N, cIn, cOut, K, hIn, wIn, hOut, wOut, groups = 18, 200, 200, 3, 29, 50, 29, 50, 1

        x = self.create_single_cpu_tensor([np.float32, 0, (N, cIn, hIn, wIn)], -5, 5)
        offset = self.create_single_cpu_tensor([np.float32, 0, (N, 2 * K * K, hOut, wOut)], -2, 2)
        mask = self.create_single_cpu_tensor([np.float32, 0, (N, K * K, hOut, wOut)], -5, 5)
        weight = self.create_single_cpu_tensor([np.float32, 0, (cOut, cIn // groups, K, K)], -5, 5) * 0.01
        npu_x, npu_offset, npu_mask, npu_weight = x.npu(), offset.npu(), mask.npu(), weight.npu()

        try:
            npu_out = modulated_deform_conv2d(npu_x, npu_offset, npu_mask, npu_weight, None, 1, 1, 1, groups).cpu()
        except ValueError as e:
            assert str(e) == "Channel only support 64-aligned"

    def test_model_cases_with_single_group(self):

        np.random.seed(42)
        N_ = [18, 18, 6, 6, 36, 36, 12, 12]
        cIn_ = [256, 512, 256, 512, 256, 512, 256, 512]
        cOut_ = [256, 512, 256, 512, 256, 512, 256, 512]
        hIn_ = [58, 29, 58, 29, 58, 29, 58, 29]
        wIn_ = [100, 50, 100, 50, 100, 50, 100, 50]
        hOut_ = [58, 29, 58, 29, 58, 29, 58, 29]
        wOut_ = [100, 50, 100, 50, 100, 50, 100, 50]
        K, groups = 3, 1

        double_benchmark_flag = False
        gpu_out_path = os.getenv("GPU_OUT_PATH", None)
        if (gpu_out_path is not None) and os.path.exists(gpu_out_path):
            double_benchmark_flag = True

        for i in range(len(N_)):

            N, cIn, cOut, hIn, wIn, hOut, wOut = N_[i], cIn_[i], cOut_[i], hIn_[i], wIn_[i], hOut_[i], wOut_[i]

            if double_benchmark_flag:

                gpu_input = torch.load(os.path.join(gpu_out_path, "mdcn_gpu_out_2", f"gpu_input_{N}_{cIn}.pt"))
                x, offset, mask, weight = gpu_input["x"], gpu_input["offset"], gpu_input["mask"], gpu_input["weight"]

                # cpu golden
                cpu_out = self.get_fwd_golden(x, offset, mask, weight, groups)
                # npu
                npu_x, npu_offset, npu_mask, npu_weight = x.npu(), offset.npu(), mask.npu(), weight.npu()
                npu_out = modulated_deform_conv2d(npu_x, npu_offset, npu_mask, npu_weight, None, 1, 1, 1, groups).cpu()

                gpu_out = torch.load(os.path.join(gpu_out_path, "mdcn_gpu_out_2", f"gpu_output_{N}_{cIn}.pt"))
                compare = CvFusedDoubleBenchmarkAccuracyCompare([npu_out], [gpu_out], [cpu_out])
                ret = compare.run()
                assert "False" not in ret, f"Accuracy check failed for model case {i + 1} with N={N}, cIn={cIn}"

            else:
                x = self.create_single_cpu_tensor([np.float32, 0, (N, cIn, hIn, wIn)], -5, 5)
                offset = self.create_single_cpu_tensor([np.float32, 0, (N, 2 * K * K, hOut, wOut)], -2, 2)
                mask = self.create_single_cpu_tensor([np.float32, 0, (N, K * K, hOut, wOut)], -5, 5)
                weight = self.create_single_cpu_tensor([np.float32, 0, (cOut, cIn // groups, K, K)], -5, 5) * 0.01
                # cpu golden
                cpu_out = self.get_fwd_golden(x, offset, mask, weight, groups)
                # npu
                npu_x, npu_offset, npu_mask, npu_weight = x.npu(), offset.npu(), mask.npu(), weight.npu()
                npu_out = modulated_deform_conv2d(npu_x, npu_offset, npu_mask, npu_weight, None, 1, 1, 1, groups).cpu()

                self.assertRtolEqual(npu_out, cpu_out, 1e-3, 1e-3)

    def test_modulated_deformable_conv2d_multi_groups(self):
        N, cIn, cOut, K, hIn, wIn, hOut, wOut, groups = 18, 512, 512, 3, 29, 50, 29, 50, 8

        cpu_x = self.create_single_cpu_tensor([np.float32, 0, (N, cIn, hIn, wIn)], -5, 5)
        cpu_w = self.create_single_cpu_tensor([np.float32, 0, (cOut, cIn // groups, K, K)], -5, 5) * 0.01
        cpu_o = self.create_single_cpu_tensor([np.float32, 0, (N, 2 * K * K, hOut, wOut)], -2, 2)
        cpu_m = self.create_single_cpu_tensor([np.float32, 0, (N, K * K, hOut, wOut)], -5, 5)
        cpu_output = self.get_fwd_golden(cpu_x, cpu_o, cpu_m, cpu_w, groups)

        output = modulated_deform_conv2d(cpu_x.npu(), cpu_o.npu(), cpu_m.npu(), cpu_w.npu(), None, 1, 1, 1, groups)
        self.assertRtolEqual(output, cpu_output)


if __name__ == "__main__":
    run_tests()
