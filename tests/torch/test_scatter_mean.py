import torch
import numpy as np
import torch_scatter

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
import mx_driving.common


class TestScatterMeanWithArgmax(TestCase):
    def cpu_op_exec(self, src, index):
        out = torch_scatter.scatter_mean(src, index.long(), out=None, dim=0)
        out = out.detach().numpy()
        return out

    def npu_op_exec(self, src, index):
        out = mx_driving.common.scatter_mean(src, index, out=None, dim=0)
        out = out.cpu()
        out = out.detach().numpy()
        return out

    def test_scatter_mean_dim2(self):
        input_list = [[[1136731, 16], [1136731, ], 100], 
                      [[200, 4000], [200, ], 100], 
                      [[1024, 99], [1024, 99], 100]]

        for input_info in input_list:
            src_shape = input_info[0]
            index_shape = input_info[1]
            index_max = input_info[2]
            for dim in range(len(index_shape)):
                cpu_src, npu_src = create_common_tensor(["float32", 2, src_shape], 0, 100)
                cpu_index, npu_index = create_common_tensor(["int32", 2, index_shape], 0, index_max)
                cpu_output = torch_scatter.scatter_mean(cpu_src, cpu_index.long(), dim=dim)
                npu_output = mx_driving.common.scatter_mean(npu_src, npu_index, None, dim)
                print(npu_output.shape)
                self.assertRtolEqual(cpu_output, npu_output.cpu())

    def test_scatter_mean_dim3(self):
        input_list = [
                        [[200, 500, 128], [200, ], 100], 
                        [[200, 5, 128], [200, 5], 100], 
                        [[200, 5, 128], [200, 5, 128], 100]
                     ]

        for input_info in input_list:
            src_shape = input_info[0]
            index_shape = input_info[1]
            index_max = input_info[2]
            for dim in range(len(index_shape)):
                cpu_src, npu_src = create_common_tensor(["float32", 2, src_shape], 0, 100)
                cpu_index, npu_index = create_common_tensor(["int32", 2, index_shape], 0, index_max)
                cpu_output = torch_scatter.scatter_mean(cpu_src, cpu_index.long(), dim=dim)
                npu_output = mx_driving.common.scatter_mean(npu_src, npu_index, None, dim)
                print(npu_output.shape)
                self.assertRtolEqual(cpu_output, npu_output.cpu())

    def test_scatter_mean_dim_more(self):
        input_list = [
                        [[200, 2, 5, 128], [200, 2], [300, 2, 5, 128], 20],
                        [[200, 1, 3, 5, 1299], [200, 1, 3], [100, 1, 3, 5, 1299], 100],
                        [[500, 20, 8, 5, 1, 16], [500, 20, 8], [500, 20, 8, 5, 1, 16], 1500]
                     ]

        for input_info in input_list:
            src_shape = input_info[0]
            index_shape = input_info[1]
            out_shape = input_info[2]
            index_max = input_info[3]
            dim = 0
            cpu_src, npu_src = create_common_tensor(["float32", 2, src_shape], 0, 100)
            cpu_index, npu_index = create_common_tensor(["int32", 2, index_shape], 0, index_max)
            cpu_output = torch_scatter.scatter_mean(cpu_src, cpu_index.long(), dim=dim)
            npu_output = mx_driving.common.scatter_mean(npu_src, npu_index, None, dim)
            print(npu_output.shape)
            self.assertRtolEqual(cpu_output, npu_output.cpu())

    def test_scatter_mean_without(self):
        input_list = [
                        [[16, 500, 128], [16, ], [16, 500, 128], 100],
                        [[16, 1, 3, 5, 1299], [16, 1, 3], [16, 1, 3, 5, 1299], 100],
                        [[256, 20, 30, 5, 1, 16], [256, 20, 30], [256, 20, 30, 5, 1, 16], 256]
                     ]

        for input_info in input_list:
            src_shape = input_info[0]
            index_shape = input_info[1]
            out_shape = input_info[2]
            dim = 0
            cpu_src, npu_src = create_common_tensor(["float32", 2, src_shape], 0, 100)
            cpu_index, npu_index = create_common_tensor(["int32", 2, index_shape], 0, index_shape[dim])
            cpu_out, npu_out = create_common_tensor(["float32", 2, out_shape], 0, 100)
            cpu_output2 = torch_scatter.scatter_mean(cpu_src, cpu_index.long(), out=cpu_out, dim=dim)
            npu_output2 = mx_driving.common.scatter_mean(npu_src, npu_index, npu_out, dim)
            self.assertRtolEqual(cpu_output2, npu_output2.cpu())
    
if __name__ == "__main__":
    run_tests()