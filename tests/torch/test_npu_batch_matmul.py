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


def gen_former_npu_outputs(projection_mat, pts_extend):
    points_2d_mm = torch.matmul(projection_mat[:, :, None, None], pts_extend[:, None, ..., None])
    return points_2d_mm


class TestBatchMatmul(TestCase):  
    def test_npu_batch_matmul_sixdim(self, device="npu"):
        projection_mat, pts_extend = gen_inputs([6, 6, 4, 4], [6, 1220, 13, 4], np.float32)
        projection_mat_fused = projection_mat.detach()
        pts_extend2_fused = pts_extend.detach()
        projection_mat.requires_grad = True
        pts_extend.requires_grad = True      
        former_npu_result = gen_former_npu_outputs(projection_mat, pts_extend)
        grad = torch.ones_like(former_npu_result)
        former_npu_result.backward(grad)
        x_grad_former_npu = projection_mat.grad
        w_grad_former_npu = pts_extend.grad

        projection_mat_fused = projection_mat_fused[:, :, None, None].contiguous()
        pts_extend2_fused = pts_extend2_fused[:, None, :, :, None, :].contiguous()
        projection_mat_fused.requires_grad = True
        pts_extend2_fused.requires_grad = True        
        result = mx_driving.npu_batch_matmul(projection_mat_fused, pts_extend2_fused)
        grad = torch.ones_like(result)
        result.backward(grad)
        x_grad_npu = projection_mat_fused.grad
        w_grad_npu = pts_extend2_fused.grad

        self.assertRtolEqual(result.detach().cpu().numpy(), former_npu_result.detach().cpu().numpy())
        self.assertRtolEqual(x_grad_former_npu.cpu().numpy(), x_grad_npu.squeeze().cpu().numpy())
        self.assertRtolEqual(w_grad_former_npu.cpu().numpy(), w_grad_npu.squeeze().cpu().numpy())
    
    def test_npu_batch_matmul_four_dim(self, device="npu"):
        projection_mat, pts_extend = gen_inputs([6, 1, 4, 4], [6, 1220, 4, 1], np.float32)
        projection_mat_fused = projection_mat.detach()
        pts_extend2_fused = pts_extend.detach()
        projection_mat.requires_grad = True
        pts_extend.requires_grad = True
        former_npu_result = torch.matmul(projection_mat, pts_extend)
        grad = torch.ones_like(former_npu_result)
        former_npu_result.backward(grad)
        x_grad_former_npu = projection_mat.grad
        w_grad_former_npu = pts_extend.grad

        projection_mat_fused.requires_grad = True
        pts_extend2_fused.requires_grad = True
        pts_extend2_fused_ = pts_extend2_fused.transpose(3, 2).contiguous()
        result = mx_driving.npu_batch_matmul(projection_mat_fused, pts_extend2_fused_)
        grad = torch.ones_like(result)
        result.backward(grad)
        x_grad_npu = projection_mat_fused.grad
        w_grad_npu = pts_extend2_fused.grad

        self.assertRtolEqual(result.detach().cpu().numpy(), former_npu_result.detach().cpu().numpy())
        self.assertRtolEqual(x_grad_former_npu.cpu().numpy(), x_grad_npu.cpu().numpy())
        self.assertRtolEqual(w_grad_former_npu.cpu().numpy(), w_grad_npu.cpu().numpy())
    
    def test_npu_batch_matmul_none_brodcast(self, device="npu"):
        projection_mat, pts_extend = gen_inputs([6, 1220, 4, 4], [6, 1220, 4, 1], np.float32)
        projection_mat_fused = projection_mat.detach()
        pts_extend2_fused = pts_extend.detach()
        projection_mat.requires_grad = True
        pts_extend.requires_grad = True      
        former_npu_result = torch.matmul(projection_mat, pts_extend)
        grad = torch.ones_like(former_npu_result)
        former_npu_result.backward(grad)
        x_grad_former_npu = projection_mat.grad
        w_grad_former_npu = pts_extend.grad

        projection_mat_fused.requires_grad = True
        pts_extend2_fused.requires_grad = True 
        pts_extend2_fused_ = pts_extend2_fused.transpose(3, 2).contiguous()
        result = mx_driving.npu_batch_matmul(projection_mat_fused, pts_extend2_fused_)
        grad = torch.ones_like(result)
        result.backward(grad)
        x_grad_npu = projection_mat_fused.grad
        w_grad_npu = pts_extend2_fused.grad

        self.assertRtolEqual(result.detach().cpu().numpy(), former_npu_result.detach().cpu().numpy())
        self.assertRtolEqual(x_grad_former_npu.cpu().numpy(), x_grad_npu.cpu().numpy())
        self.assertRtolEqual(w_grad_former_npu.cpu().numpy(), w_grad_npu.cpu().numpy())
    
    def test_npu_batch_matmul_need_brodcast(self, device="npu"):
        projection_mat, pts_extend = gen_inputs([1, 1, 4, 4], [6, 1220, 4, 1], np.float32)
        projection_mat_fused = projection_mat.detach()
        pts_extend2_fused = pts_extend.detach()
        projection_mat.requires_grad = True
        pts_extend.requires_grad = True      
        former_npu_result = torch.matmul(projection_mat, pts_extend)
        grad = torch.ones_like(former_npu_result)
        former_npu_result.backward(grad)
        x_grad_former_npu = projection_mat.grad
        w_grad_former_npu = pts_extend.grad

        projection_mat_fused.requires_grad = True
        pts_extend2_fused.requires_grad = True  
        pts_extend2_fused_ = pts_extend2_fused.transpose(3, 2).contiguous()   
        result = mx_driving.npu_batch_matmul(projection_mat_fused, pts_extend2_fused_)
        grad = torch.ones_like(result)
        result.backward(grad)
        x_grad_npu = projection_mat_fused.grad
        w_grad_npu = pts_extend2_fused.grad

        self.assertRtolEqual(result.detach().cpu().numpy(), former_npu_result.detach().cpu().numpy())
        self.assertRtolEqual(x_grad_former_npu.cpu().numpy(), x_grad_npu.cpu().numpy())
        self.assertRtolEqual(w_grad_former_npu.cpu().numpy(), w_grad_npu.cpu().numpy())
    
    def test_npu_batch_matmul_kernel_3(self, device="npu"):
        projection_mat, pts_extend = gen_inputs([6, 6, 3, 3], [6, 1220, 13, 3], np.float32)
        projection_mat_fused = projection_mat.detach()
        pts_extend2_fused = pts_extend.detach()
        projection_mat.requires_grad = True
        pts_extend.requires_grad = True      
        former_npu_result = gen_former_npu_outputs(projection_mat, pts_extend)
        grad = torch.ones_like(former_npu_result)
        former_npu_result.backward(grad)
        x_grad_former_npu = projection_mat.grad
        w_grad_former_npu = pts_extend.grad

        projection_mat_fused = projection_mat_fused[:, :, None, None].contiguous()
        pts_extend2_fused = pts_extend2_fused[:, None, :, :, None, :].contiguous()
        projection_mat_fused.requires_grad = True
        pts_extend2_fused.requires_grad = True        
        result = mx_driving.npu_batch_matmul(projection_mat_fused, pts_extend2_fused)
        grad = torch.ones_like(result)
        result.backward(grad)
        x_grad_npu = projection_mat_fused.grad
        w_grad_npu = pts_extend2_fused.grad

        self.assertRtolEqual(result.detach().cpu().numpy(), former_npu_result.detach().cpu().numpy())
        self.assertRtolEqual(x_grad_former_npu.cpu().numpy(), x_grad_npu.squeeze().cpu().numpy())
        self.assertRtolEqual(w_grad_former_npu.cpu().numpy(), w_grad_npu.squeeze().cpu().numpy())
    
    def test_npu_batch_matmul_kernel_3_dim_4(self, device="npu"):
        projection_mat, pts_extend = gen_inputs([6, 1, 3, 3], [6, 1220, 3, 1], np.float32)
        projection_mat_fused = projection_mat.detach()
        pts_extend2_fused = pts_extend.detach()
        projection_mat.requires_grad = True
        pts_extend.requires_grad = True      
        former_npu_result = torch.matmul(projection_mat, pts_extend)
        grad = torch.ones_like(former_npu_result)
        former_npu_result.backward(grad)
        x_grad_former_npu = projection_mat.grad
        w_grad_former_npu = pts_extend.grad

        projection_mat_fused.requires_grad = True
        pts_extend2_fused.requires_grad = True
        pts_extend2_fused_ = pts_extend2_fused.transpose(3, 2).contiguous()      
        result = mx_driving.npu_batch_matmul(projection_mat_fused, pts_extend2_fused_)
        grad = torch.ones_like(result)
        result.backward(grad)
        x_grad_npu = projection_mat_fused.grad
        w_grad_npu = pts_extend2_fused.grad

        self.assertRtolEqual(result.detach().cpu().numpy(), former_npu_result.detach().cpu().numpy())
        self.assertRtolEqual(x_grad_former_npu.cpu().numpy(), x_grad_npu.cpu().numpy())
        self.assertRtolEqual(w_grad_former_npu.cpu().numpy(), w_grad_npu.cpu().numpy())

if __name__ == "__main__":
    run_tests()