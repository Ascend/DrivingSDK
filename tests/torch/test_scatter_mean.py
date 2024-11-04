import torch
import numpy as np
import torch_scatter

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
import mx_driving.common


class TestScatterMeanWithArgmax(TestCase):
    def cpu_op_exec(self, src, index, out=None, dim=0, dim_size=None):
        src.requires_grad = True
        out = torch_scatter.scatter_mean(src, index.long(), out=out, dim=dim, dim_size=dim_size)
        out.backward(out)
        grad_in = src.grad
        return out, grad_in

    def npu_op_exec(self, src, index, out=None, dim=0, dim_size=None):
        src.requires_grad = True
        out = mx_driving.common.scatter_mean(src, index, out, dim, dim_size)
        out.backward(out)
        grad_in = src.grad
        return out.cpu(), grad_in.cpu()


    def test_scatter_mean_dim2(self):
        input_list = [[[1136731, 16], [1136731, ], 100], 
                      [[200, 4000], [200, ], 100], 
                      [[1024, 99], [1024, 99], 100],
                      [[1, 1], [1, 1], 2]]

        for input_info in input_list:
            src_shape = input_info[0]
            index_shape = input_info[1]
            index_max = input_info[2]
            for dim in range(len(index_shape)):
                cpu_src, npu_src = create_common_tensor(["float32", 2, src_shape], 0, 100)
                cpu_index, npu_index = create_common_tensor(["int32", 2, index_shape], 0, index_max)
                cpu_output, cpu_grad_in = self.cpu_op_exec(cpu_src, cpu_index.long(), dim=dim)
                npu_output, npu_grad_in = self.npu_op_exec(npu_src, npu_index, None, dim)

                self.assertRtolEqual(cpu_output, npu_output)
                self.assertRtolEqual(cpu_grad_in, npu_grad_in)

    def test_scatter_mean_dim3(self):
        input_list = [
                        [[200, 500, 128], [200, ], 100], 
                        [[200, 5, 128], [200, 5], 100], 
                        [[3, 5, 8], [3, 5, 8], 100]
                     ]

        for input_info in input_list:
            src_shape = input_info[0]
            index_shape = input_info[1]
            index_max = input_info[2]
            for dim in range(len(index_shape)):
                cpu_src, npu_src = create_common_tensor(["float32", 2, src_shape], 0, 100)
                cpu_index, npu_index = create_common_tensor(["int32", 2, index_shape], 0, index_max)
                cpu_output, cpu_grad_in = self.cpu_op_exec(cpu_src, cpu_index.long(), dim=dim)
                npu_output, npu_grad_in = self.npu_op_exec(npu_src, npu_index, None, dim)

                self.assertRtolEqual(cpu_output, npu_output)
                self.assertRtolEqual(cpu_grad_in, npu_grad_in)

    def test_scatter_mean_dim_more(self):
        input_list = [
                        [[200, 2, 5, 128], [200, 2], [300, 2, 5, 128], 20],
                        [[200, 1, 3, 5, 1299], [200, 1, 3], [100, 1, 3, 5, 1299], 100],
                        [[500, 20, 8, 5, 1, 16], [500, 20, 8], [500, 20, 8, 5, 1, 16], 800]
                     ]

        for input_info in input_list:
            src_shape = input_info[0]
            index_shape = input_info[1]
            out_shape = input_info[2]
            index_max = input_info[3]
            for dim in range(len(index_shape)):
                cpu_src, npu_src = create_common_tensor(["float32", 2, src_shape], 0, 100)
                cpu_index, npu_index = create_common_tensor(["int32", 2, index_shape], 0, index_max)
                cpu_output, cpu_grad_in = self.cpu_op_exec(cpu_src, cpu_index.long(), dim=dim)
                npu_output, npu_grad_in = self.npu_op_exec(npu_src, npu_index, None, dim)

                self.assertRtolEqual(cpu_output, npu_output)
                self.assertRtolEqual(cpu_grad_in, npu_grad_in)

    def test_scatter_mean_without(self):
        input_list = [
                        [[16, 500, 128], [16, ], [10, 500, 128], 0],
                        [[16, 1, 3, 5, 1299], [16, 1, 3], [16, 4, 3, 5, 1299], 1],
                        [[16, 1, 3, 5, 1299], [16, 1, 3], [16, 1, 3, 5, 1299], 2],
                        [[256, 20, 30, 5, 1, 16], [256, 20, 30], [256, 20, 10, 5, 1, 16], 2],
                        [[1, 1, 1], [1, 1, 1], [1, 3, 1], 1]
                     ]

        for input_info in input_list:
            src_shape = input_info[0]
            index_shape = input_info[1]
            out_shape = input_info[2]
            dim = input_info[3]

            cpu_src, npu_src = create_common_tensor(["float32", 2, src_shape], 0, 100)
            cpu_index, npu_index = create_common_tensor(["int32", 2, index_shape], 0, out_shape[dim])
            cpu_out, npu_out = create_common_tensor(["float32", 2, out_shape], 0, 100)
            cpu_output, cpu_grad_in = self.cpu_op_exec(cpu_src, cpu_index.long(), out=cpu_out, dim=dim)
            npu_output, npu_grad_in = self.npu_op_exec(npu_src, npu_index, out=npu_out, dim=dim)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad_in, npu_grad_in)
    
    def test_scatter_mean_with_dimsize(self):
        input_list = [
                        [[16, 5, 128], [16, 5, 128], [16, 5, 128], 100],
                        [[16, 2, 5, 128], [16, 2], [16, 2, 5, 128], 100],
                        [[256, 1, 30, 5, 1, 16], [256, 1, 30], [256, 1, 30, 5, 1, 16], 256]
                     ]

        for input_info in input_list:
            src_shape = input_info[0]
            index_shape = input_info[1]
            out_shape = input_info[2]
            dim_size = input_info[3]
            for dim in range(len(index_shape)):
                cpu_src, npu_src = create_common_tensor(["float32", 2, src_shape], 0, 100)
                cpu_index, npu_index = create_common_tensor(["int32", 2, index_shape], 0, dim_size)
                cpu_out, npu_out = create_common_tensor(["float32", 2, out_shape], 0, 100)
                cpu_output, cpu_grad_in = self.cpu_op_exec(cpu_src, cpu_index.long(), out=None, dim=dim, dim_size=dim_size)
                npu_output, npu_grad_in = self.npu_op_exec(npu_src, npu_index, out=None, dim=dim, dim_size=dim_size)
                self.assertRtolEqual(cpu_output, npu_output)
                self.assertRtolEqual(cpu_grad_in, npu_grad_in)
    
if __name__ == "__main__":
    run_tests()