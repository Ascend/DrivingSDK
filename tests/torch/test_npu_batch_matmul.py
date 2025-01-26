import unittest

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests
import mx_driving

@golden_data_cache(__file__)
def gen_inputs(shape1, shape2, dtype):
    projection_mat =torch.randn(shape1).npu()
    pts_extend =torch.randn(shape2).npu()
    return projection_mat, pts_extend


@golden_data_cache(__file__)
def gen_former_npu_outputs(projection_mat, pts_extend):
    points_2d_mm = torch.matmul(projection_mat[:, :, None, None], pts_extend[:, None, ..., None])
    return points_2d_mm


class TestBatchMatmul(TestCase):  
    def test_npu_batch_matmul(self, device="npu"):
        projection_mat, pts_extend = gen_inputs([6, 6, 4, 4], [6, 1220, 13, 4], np.float32)
        projection_mat_fused = projection_mat.detach()
        pts_extend2_fused = pts_extend.detach()
        projection_mat.requires_grad = True
        pts_extend.requires_grad = True      
        former_npu_result = gen_former_npu_outputs(projection_mat, pts_extend)
        grad = torch.ones_like(former_npu_result)
        former_npu_result.backward(grad)  

        projection_mat_fused = projection_mat_fused[:, :, None, None].contiguous()
        pts_extend2_fused = pts_extend2_fused[:, None, :, :, None, :].contiguous()
        projection_mat_fused.requires_grad = True
        pts_extend2_fused.requires_grad = True        
        result = mx_driving.npu_batch_matmul(projection_mat_fused, pts_extend2_fused)
        grad = torch.ones_like(result)
        result.backward(grad)
        print("result", result.shape)
        print("former_npu_result", former_npu_result.shape)

        self.assertRtolEqual(result.detach().cpu().numpy(), former_npu_result.detach().cpu().numpy())
if __name__ == "__main__":
    run_tests()