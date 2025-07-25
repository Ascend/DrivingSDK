from collections import namedtuple
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests
import mx_driving
from mx_driving import scatter_add, scatter_max

torch.manual_seed(1)


@golden_data_cache(__file__)
def golden_graph_softmax(src, index):
    N = torch.max(index) + 1
    src_max = scatter_max(src.detach(), index, None)[0]
    out = src - src_max.index_select(0, index)
    out = out.exp()
    out_sum = scatter_add(out, index, None, 0, N) + 1e-16
    out_sum = out_sum.index_select(0, index)
    return out / out_sum


@golden_data_cache(__file__)
def golden_graph_softmax_grad(index, softmax_out, grad_out):
    N = torch.max(index) + 1
    grad_out = softmax_out * grad_out
    grad_sum = scatter_add(grad_out, index, None, 0, N)
    grad_sum = grad_sum.index_select(0, index)
    grad_src = grad_out - softmax_out * grad_sum
    return grad_src


@golden_data_cache(__file__)
def gen_inputs(Num_Edge, Num_Feature, data_range):
    src = (torch.rand((Num_Edge, Num_Feature)) - 0.5) * 2 * data_range
    if Num_Edge == 1112:
        index = torch.zeros(Num_Edge,) # test for multiple edges pointing to the same node
    else:
        index = torch.arange(0, 1500000 + 1500000 // Num_Edge, 1500000 // Num_Edge)[:Num_Edge] # iterate through the range of index, [0, 1500000)
    grad_out = (torch.rand((Num_Edge, Num_Feature)) * 1e-3).float()
    return src, index, grad_out


def golden_to_exec(src, index, grad_out):
    golden_src = src.npu()
    golden_index = index.int().npu()
    golden_grad_out = grad_out.npu()

    golden_src.requires_grad_()
    golden_output = golden_graph_softmax(golden_src, golden_index)
    golden_src_grad = golden_graph_softmax_grad(golden_index, golden_output, golden_grad_out)
    return golden_output, golden_src_grad.float()


def npu_to_exec(src, index, grad_out):
    npu_src = src.npu()
    npu_index = index.npu()
    npu_grad_out = grad_out.npu()

    npu_src.requires_grad_()
    npu_output = mx_driving.graph_softmax(npu_src, npu_index)
    npu_output.backward(npu_grad_out)
    return npu_output, npu_src.grad.float()


class TestGraphSoftmax(TestCase):
    def test_graph_softmax(self):
        Num_Feature = 8 # Feature number is 8 in QCNet Model
        data_range = 500 # iterate through the range of src, [-500, 500)
        Num_Edge_List = [i for i in range(1, 50000, 1111)] # iterate through the range of Num_Edge, [1, 50000)
        Num_Edge_List.append(50000) # test for max Num_Edge and max index value

        for Num_Edge in Num_Edge_List:
            src, index, grad_out = gen_inputs(Num_Edge, Num_Feature, data_range)
            golden_output, golden_src_grad = golden_to_exec(src, index, grad_out)
            npu_output, npu_src_grad = npu_to_exec(src, index, grad_out)
            self.assertRtolEqual(golden_output, npu_output)
            self.assertRtolEqual(golden_src_grad, npu_src_grad)

if __name__ == "__main__":
    run_tests()