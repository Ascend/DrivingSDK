import numpy as np
import torch
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving


class TestThreeinterpolate(TestCase):
    @golden_data_cache(__file__)
    def cpu_op_exec(self, feat, idx, wt):
        bs, cs, ms = feat.shape
        ns = idx.shape[1]
        out = np.zeros((bs, cs, ns)).astype(feat.dtype)
        # forward
        for b in range(bs):
            for c in range(cs):
                for n in range(ns):
                    out[b, c, n] = feat[b, c, idx[b, n, 0]] * wt[b, n, 0] + \
                                    feat[b, c, idx[b, n, 1]] * wt[b, n, 1] + \
                                    feat[b, c, idx[b, n, 2]] * wt[b, n, 2]
        
        grad_out = np.zeros((bs, cs, ms)).astype(feat.dtype)
        grad_out = grad_out.transpose(0, 2, 1)
        for b in range(bs):
            for n in range(ns):
                ind = idx[b, n, :]
                weight = wt[b, n, :]
                grad_out[b, ind[0], :] += weight[0].repeat(cs)
                grad_out[b, ind[1], :] += weight[1].repeat(cs)
                grad_out[b, ind[2], :] += weight[2].repeat(cs)
        grad_out = grad_out.transpose(0, 2, 1)
        
        return out, grad_out
    
    def npu_op_exec(self, feat, idx, wt):
        feat.requires_grad = True
        
        out = mx_driving.three_interpolate(feat, idx, wt)
        out.backward(torch.ones_like(out))
        grad_out = feat.grad
        grad_out = grad_out.detach().cpu().numpy()
        out = out.detach().cpu().numpy()
        
        return out, grad_out
    
    def test_three_interpolate_with_grad(self):
        bs = [2, 10, 224]
        cs = [3, 20, 45]
        ms = [4, 17, 224]
        ns = [5, 34, 150]
        for i in range(3):
            np.random.seed(i)
            features = np.random.uniform(-1000, 1000, size=(bs[i], cs[i], ms[i])).astype(np.float32)
            indices = np.random.randint(0, ms[i], size=(bs[i], ns[i], 3)).astype(np.int32)
            weights = np.random.uniform(0, 1, size=(bs[i], ns[i], 3)).astype(np.float32)
            
            npu_features = torch.from_numpy(features).to(torch.float32).npu()
            npu_indices = torch.from_numpy(indices).int().npu()
            npu_weights = torch.from_numpy(weights).to(torch.float32).npu()
            cpu_output = self.cpu_op_exec(features, indices, weights)
            npu_output = self.npu_op_exec(npu_features, npu_indices, npu_weights)
            self.assertRtolEqual(cpu_output[0], npu_output[0])
            self.assertRtolEqual(cpu_output[1], npu_output[1])
        
        
if __name__ == "__main__":
    run_tests()