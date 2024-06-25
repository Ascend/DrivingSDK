import unittest
import torch
import numpy as np
from torch_scatter import scatter_mean

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import mx_driving.common


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestScatterMeanGradFunction(TestCase):
    def gen_data(self, x_shape, index_shape, dim):
        index = np.random.uniform(0, x_shape[dim], tuple(index_shape)).astype(np.int32)
        x = np.random.uniform(-10, 10, tuple(x_shape)).astype(np.float32)
        x_tensor = torch.from_numpy(x)
        index_tensor = torch.from_numpy(index)
        
        return [x_tensor, index_tensor]

    def cpu_to_exec(self, x_tensor, index_tensor, dim):
        x_tensor.requires_grad_(True)
        output = scatter_mean(x_tensor, index_tensor.to(torch.int64), dim)
        grad_out_tensor = torch.ones_like(output)
        output.backward(grad_out_tensor)
        result_cpu = x_tensor.grad
        return result_cpu.numpy(), grad_out_tensor

    def npu_to_exec(self, index_tensor, grad_out_tensor, dim):
        result_npu = mx_driving.common.npu_scatter_mean_grad(grad_out_tensor.npu(),
                                                      index_tensor.to(torch.int32).npu(),
                                                      dim)
        return result_npu.cpu().numpy()

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `ScatterMeanGrad` is only supported on 910B, skip this ut!")
    def test_scatter_mean_grad_function(self):
        x_shape_list = [[32, 60, 15], [32, 41], [1000, 33], [1089, 49], [2, 8, 80], [10, 49, 80]]
        index_shape_list = [[32, 60, 15], [32, 41], [1000, 33], [1089], [2, 8], [10, 49]]
        dim_list = [1, 0, -1, 0, 1, 1]

        for x_shape, index_shape, dim in zip(x_shape_list, index_shape_list, dim_list):
            x_tensor, index_tensor = self.gen_data(x_shape, index_shape, dim)
            result_cpu, grad_out_tensor = self.cpu_to_exec(x_tensor, index_tensor, dim)
            result_npu = self.npu_to_exec(index_tensor, grad_out_tensor, dim)

            self.assertRtolEqual(result_cpu, result_npu)

if __name__ == '__main__':
    run_tests()