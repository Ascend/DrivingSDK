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
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def get_cpu_golden(self, dtype, x, offset, mask, weight, groups, grad_out):
        x_cpu = x.detach().clone().cpu().to(dtype)
        offset_cpu = offset.detach().clone().cpu().to(dtype)
        mask_cpu = mask.detach().clone().cpu().to(dtype)
        weight_cpu = weight.detach().clone().cpu().to(dtype)
        x_cpu.grad, offset_cpu.grad, mask_cpu.grad, weight_cpu.grad = None, None, None, None
        
        x_cpu.requires_grad = True
        offset_cpu.requires_grad = True
        mask_cpu.requires_grad = True
        weight_cpu.requires_grad = True
        
        out = mmcv_modulated_deform_conv2d(x_cpu, offset_cpu, mask_cpu, weight_cpu, None, 1, 1, 1, groups)
        out.backward(grad_out, retain_graph=True)

        return x_cpu.grad, offset_cpu.grad, mask_cpu.grad, weight_cpu.grad
    
    def get_npu_output(self, x, offset, mask, weight, groups, grad_out):
        x_npu = x.detach().clone().npu()
        offset_npu = offset.detach().clone().npu()
        mask_npu = mask.detach().clone().npu()
        weight_npu = weight.detach().clone().npu()
        x_npu.grad, offset_npu.grad, mask_npu.grad, weight_npu.grad = None, None, None, None
        
        x_npu.requires_grad = True
        offset_npu.requires_grad = True
        mask_npu.requires_grad = True
        weight_npu.requires_grad = True
        
        out = modulated_deform_conv2d(x_npu, offset_npu, mask_npu, weight_npu, None, 1, 1, 1, groups)
        
        out.backward(grad_out.npu(), retain_graph=True)
        return x_npu.grad.cpu(), offset_npu.grad.cpu(), mask_npu.grad.cpu(), weight_npu.grad.cpu()

    def single_check_result(self, npu_out, cpu_out):
        x_grad_cpu, offset_grad_cpu, mask_grad_cpu, weight_grad_cpu = cpu_out
        x_grad_npu, offset_grad_npu, mask_grad_npu, weight_grad_npu = npu_out
        
        self.assertRtolEqual(x_grad_npu, x_grad_cpu, 1e-3, 1e-3)
        self.assertRtolEqual(offset_grad_npu, offset_grad_cpu, 1e-3, 1e-3)
        self.assertRtolEqual(mask_grad_npu, mask_grad_cpu, 1e-3, 1e-3)
        self.assertRtolEqual(weight_grad_npu, weight_grad_cpu, 1e-1, 1e-1)
        
    def double_check_result(self, file_name, npu_out, cpu_out, gpu_data):
        x_grad_cpu, offset_grad_cpu, mask_grad_cpu, weight_grad_cpu = cpu_out
        x_grad_npu, offset_grad_npu, mask_grad_npu, weight_grad_npu = npu_out
        
        x_grad_gpu, offset_grad_gpu, mask_grad_gpu, weight_grad_gpu = \
            gpu_data["x_grad"], gpu_data["offset_grad"], gpu_data["mask_grad"], gpu_data["weight_grad"]
        
        compare = CvFusedDoubleBenchmarkAccuracyCompare([x_grad_npu, offset_grad_npu, mask_grad_npu, weight_grad_npu],
                                                        [x_grad_gpu, offset_grad_gpu, mask_grad_gpu, weight_grad_gpu],
                                                        [x_grad_cpu, offset_grad_cpu, mask_grad_cpu, weight_grad_cpu])
        ret = compare.run()
        assert "False" not in ret, f"Accuracy check failed for model case {file_name}"
    
    def test_bevformer_model_case(self):
        gpu_out_path = os.getenv("GPU_OUT_PATH", None)
        double_benchmark_flag = (gpu_out_path is not None) and os.path.exists(gpu_out_path)
        name = "test_bevformer_model_case"
        
        cases = [
            [6, 512, 512, 3, 29, 50, 29, 50, 1],
            [6, 256, 256, 3, 58, 100, 58, 100, 1],
            [12, 512, 512, 3, 29, 50, 29, 50, 1],
            [12, 256, 256, 3, 58, 100, 58, 100, 1],
        ]
        
        for i, case in enumerate(cases):
            N, cIn, cOut, K, hIn, wIn, hOut, wOut, groups = case
            if double_benchmark_flag:
                file_name = f"{name}_{i}.pt"
                path = os.path.join(gpu_out_path, "modulated_deformable_conv2d_grad", file_name)
                gpu_data = torch.load(path, map_location="cpu")
                
                x, offset, mask, weight, grad_out = \
                    gpu_data["x"], gpu_data["offset"], gpu_data["mask"], gpu_data["weight"], gpu_data["grad_out"]
                    
                cpu_out = self.get_cpu_golden(torch.float64, x, offset, mask, weight, groups, grad_out)
                npu_out = self.get_npu_output(x, offset, mask, weight, groups, grad_out)
                
                self.double_check_result(file_name, npu_out, cpu_out, gpu_data)
            else:
                x = self.create_single_cpu_tensor([np.float32, 0, (N, cIn, hIn, wIn)], -5, 5)
                offset = self.create_single_cpu_tensor([np.float32, 0, (N, 2 * K * K, hOut, wOut)], -2, 2)
                mask = self.create_single_cpu_tensor([np.float32, 0, (N, K * K, hOut, wOut)], -5, 5)
                weight = self.create_single_cpu_tensor([np.float32, 0, (cOut, cIn // groups, K, K)], -5, 5) * 0.001
                grad_out = self.create_single_cpu_tensor([np.float32, 0, (N, cIn, hOut, wOut)], -5, 5)

                cpu_out = self.get_cpu_golden(torch.float32, x, offset, mask, weight, groups, grad_out)
                npu_out = self.get_npu_output(x, offset, mask, weight, groups, grad_out)
                self.single_check_result(npu_out, cpu_out)
        
if __name__ == "__main__":
    run_tests()
