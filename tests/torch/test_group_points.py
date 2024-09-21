import unittest
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from mx_driving.point import npu_group_points


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestGroupPoints(TestCase):
    def cpu_group_points(self, points, indices, out):
        
        B, npoints, nsample = indices.shape
        features = points.transpose(0, 2, 1)
        output = out.transpose(0, 2, 3, 1)
        for b in range(B):
            for np_ in range(npoints):
                for ns in range(nsample):
                    temp = features[b, indices[b, np_, ns], :]
                    output[b, np_, ns, :] = temp
        
        output = output.transpose(0, 3, 1, 2)
        return output

    @unittest.skipIf((DEVICE_NAME != 'Ascend910B'), "OP `GroupPoints` is only supported on 910B, skip this ut!")
    def test_group_points(self):
        
        dtype = [torch.float, torch.half]
        astype = [np.float32, np.float16]
        
        for i in range(5):
            
            np.random.seed(i)
            B = np.random.randint(1, 500)
            C = np.random.randint(1, 500)
            N = np.random.randint(1, 500)
            npoints = np.random.randint(1, 10)
            nsample = np.random.randint(1, 10)
            mean = np.random.uniform(-100, 100)
            std_dev = np.random.uniform(0, 25)

            for j in range(2):
                
                np_points = np.random.normal(mean, std_dev, (B, C, N)).astype(astype[j])
                np_indices = np.random.randint(0, N, (B, npoints, nsample)).astype(np.int32)
                np_out = np.zeros((B, C, npoints, nsample)).astype(astype[j])
                
                th_points = torch.from_numpy(np_points).npu().to(dtype[j])
                th_indices = torch.from_numpy(np_indices).int().npu()

                cpu_out = self.cpu_group_points(np_points, np_indices, np_out)
                npu_out = npu_group_points(th_points, th_indices)

                self.assertRtolEqual(cpu_out, npu_out.cpu().numpy())


if __name__ == "__main__":
    run_tests()
