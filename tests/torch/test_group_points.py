import unittest

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving
import mx_driving.point


# pylint: disable=too-many-arguments,huawei-too-many-arguments
@golden_data_cache(__file__)
def cpu_gen_inputs(B, C, N, mean, std_dev, npoints, nsample, dtype):
    
    torch_points = torch.normal(mean, std_dev, size=(B, C, N), dtype=dtype)
    torch_indices = torch.randint(0, N, size=(B, npoints, nsample), dtype=torch.int32)
    
    return torch_points, torch_indices


class TestGroupPoints(TestCase):
    
    @golden_data_cache(__file__)
    def cpu_group_points(self, points, indices):
        
        B, npoints, nsample = indices.shape
        features = points.transpose(1, 2)  # (B, N, C)

        batch_indices = torch.arange(B, device=points.device).view(B, 1, 1).expand(-1, npoints, nsample)
        output = features[batch_indices, indices, :]  # (B, npoints, nsample, C)
        output = output.permute(0, 3, 1, 2)  # (B, npoints, nsample, C)
    
        return output

    def test_group_points(self):
        
        np.random.seed(1)
        torch.manual_seed(1)
        B_lists = [2, 173, 40, 173, 173, 87]
        C_lists = [3, 47, 16, 47, 147, 11]
        N_lists = [4, 9, 10, 20, 20, 49]
        np_lists = [3, 1, 8, 8, 17, 207]
        ns_lists = [4, 4, 8, 8, 19, 81]
        dtype = [torch.float, torch.half]
        
        for i in range(len(B_lists)):
            
            B, C, N, npoints, nsample = B_lists[i], C_lists[i], N_lists[i], np_lists[i], ns_lists[i]            
            mean = np.random.uniform(-100, 100)
            std_dev = np.random.uniform(0, 25)

            for j in range(2):
                
                th_points, th_indices = cpu_gen_inputs(B, C, N, mean, std_dev, npoints, nsample, dtype[j])
                npu_points = th_points.npu()
                npu_indices = th_indices.npu()

                cpu_out = self.cpu_group_points(th_points, th_indices)
                npu_out = mx_driving.point.npu_group_points(npu_points, npu_indices)
                npu_out = mx_driving.point.group_points(npu_points, npu_indices)
                out = mx_driving.group_points(npu_points, npu_indices)
                
                self.assertRtolEqual(cpu_out, npu_out.cpu())
                self.assertRtolEqual(cpu_out, out.cpu())


if __name__ == "__main__":
    run_tests()