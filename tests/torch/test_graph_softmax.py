import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests
import mx_driving
from mx_driving import scatter_add, scatter_max

torch.manual_seed(1)


@golden_data_cache(__file__)
def golden_python(src, index):
    N = torch.max(index) + 1
    src_max = scatter_max(src.detach(), index, None)[0]
    out = src - src_max.index_select(0, index)
    out = out.exp()
    out_sum = scatter_add(out, index, None, 0, N) + 1e-16
    out_sum = out_sum.index_select(0, index)
    return out / out_sum


@golden_data_cache(__file__)
def gen_inputs(Num_Edge, Num_Feature, data_range):
    src = (torch.rand((Num_Edge, Num_Feature)) - 0.5) * 2 * data_range
    index = torch.arange(0, 1500000 + 1500000 // Num_Edge, 1500000 // Num_Edge)[:Num_Edge] # iterate through the range of index, [0, 1500000)
    return src, index


class TestGraphSoftmax(TestCase):
    def test_graph_softmax(self):
        Num_Feature = 8 # Feature number is 8 in QCNet Model
        data_range = 500 # iterate through the range of src, [-500, 500)
        Num_Edge_List = [i for i in range(1, 50000, 1111)] # iterate through the range of Num_Edge, [1, 50000)
        Num_Edge_List.append(50000) # test for max Num_Edge and max index value
        
        for Num_Edge in Num_Edge_List:
            src, index = gen_inputs(Num_Edge, Num_Feature, data_range)
            if Num_Edge == 1112:
                index = torch.zeros(Num_Edge,) # test for multiple edges pointing to the same node
            out_obj = golden_python(src.npu(), index.to(torch.int32).npu())
            out_npu = mx_driving.graph_softmax(src.npu(), index.to(torch.int32).npu())
            self.assertRtolEqual(out_obj, out_npu)
    
if __name__ == "__main__":
    run_tests()