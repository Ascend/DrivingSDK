import numpy as np
import torch
import torch_npu
import torch_scatter
from data_cache import golden_data_cache
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.testcase import TestCase, run_tests
import mx_driving


@golden_data_cache(__file__)
def cpu_gen_inputs(src_shape, index_shape, index_max):
    cpu_src = np.random.uniform(0, 100, size=src_shape).astype(np.float32)
    cpu_index = np.random.uniform(0, index_max, size=index_shape).astype(np.int32)
    cpu_src = torch.from_numpy(cpu_src)
    cpu_index = torch.from_numpy(cpu_index)
    return cpu_src, cpu_index


class TestScatterMeanWithArgmax(TestCase):
    @golden_data_cache(__file__)
    def cpu_op_exec(self, src, index, out=None, dim=0, dim_size=None):
        src.requires_grad = True
        out = torch_scatter.scatter_add(src, index.long(), out=out, dim=dim, dim_size=dim_size)
        out.backward(out)
        grad_in = src.grad
        return out, grad_in

    def npu_op_exec(self, src, index, out=None, dim=0, dim_size=None):
        src.requires_grad = True
        if out is not None:
            out_in = out.clone()
        else:
            out_in = out
        res = mx_driving.scatter_add(src, index, out_in, dim, dim_size)
        res.backward(res)
        grad_in = src.grad
        return res.cpu(), grad_in.cpu()


    def test_scatter_add_dim2(self):
        input_list = [[[1136731, 16], [1136731, ], 100], 
                      [[1024, 99], [1024, 99], 100],
                      [[5, 17], [5, ], 5],
                      [[1, 1], [1, 1], 2]]

        for input_info in input_list:
            src_shape = input_info[0]
            index_shape = input_info[1]
            index_max = input_info[2]
            for dim in range(len(index_shape)):
                cpu_src, cpu_index = cpu_gen_inputs(src_shape, index_shape, index_max)
                npu_src, npu_index = cpu_src.npu(), cpu_index.npu()

                cpu_output, cpu_grad_in = self.cpu_op_exec(cpu_src, cpu_index.long(), dim=dim)
                npu_output, npu_grad_in = self.npu_op_exec(npu_src, npu_index, None, dim)

                self.assertRtolEqual(cpu_output, npu_output)
                self.assertRtolEqual(cpu_grad_in, npu_grad_in)

    def test_scatter_add_dim3(self):
        input_list = [
                        [[200, 500, 128], [200, ], 100], 
                        [[3, 5, 8], [3, 5, 8], 100]
                     ]

        for input_info in input_list:
            src_shape = input_info[0]
            index_shape = input_info[1]
            index_max = input_info[2]
            for dim in range(len(index_shape)):
                cpu_src, cpu_index = cpu_gen_inputs(src_shape, index_shape, index_max)
                npu_src, npu_index = cpu_src.npu(), cpu_index.npu()
                cpu_output, cpu_grad_in = self.cpu_op_exec(cpu_src, cpu_index.long(), dim=dim)
                npu_output, npu_grad_in = self.npu_op_exec(npu_src, npu_index, None, dim)

                self.assertRtolEqual(cpu_output, npu_output)
                self.assertRtolEqual(cpu_grad_in, npu_grad_in)

    def test_scatter_add_dim_more(self):
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
                cpu_src, cpu_index = cpu_gen_inputs(src_shape, index_shape, index_max)
                npu_src, npu_index = cpu_src.npu(), cpu_index.npu()
                cpu_output, cpu_grad_in = self.cpu_op_exec(cpu_src, cpu_index.long(), dim=dim)
                npu_output, npu_grad_in = self.npu_op_exec(npu_src, npu_index, None, dim)

                self.assertRtolEqual(cpu_output, npu_output)
                self.assertRtolEqual(cpu_grad_in, npu_grad_in)

    def test_scatter_add_without(self):
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

            cpu_src, cpu_index = cpu_gen_inputs(src_shape, index_shape, out_shape[dim])
            npu_src, npu_index = cpu_src.npu(), cpu_index.npu()
            cpu_out, npu_out = create_common_tensor(["float32", 2, out_shape], 0, 100)
            cpu_output, cpu_grad_in = self.cpu_op_exec(cpu_src, cpu_index.long(), out=cpu_out, dim=dim)
            npu_output, npu_grad_in = self.npu_op_exec(npu_src, npu_index, out=npu_out, dim=dim)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad_in, npu_grad_in)
    
    def test_scatter_add_with_dimsize(self):
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
                cpu_src, cpu_index = cpu_gen_inputs(src_shape, index_shape, dim_size)
                npu_src, npu_index = cpu_src.npu(), cpu_index.npu()
                cpu_output, cpu_grad_in = self.cpu_op_exec(cpu_src, cpu_index.long(), out=None, dim=dim, dim_size=dim_size)
                npu_output, npu_grad_in = self.npu_op_exec(npu_src, npu_index, out=None, dim=dim, dim_size=dim_size)
                self.assertRtolEqual(cpu_output, npu_output)
                self.assertRtolEqual(cpu_grad_in, npu_grad_in)
    
if __name__ == "__main__":
    run_tests()