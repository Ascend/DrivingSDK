import unittest

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving._C
import mx_driving.detection


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestRoIAwarePool3dGrad(TestCase):
    @golden_data_cache(__file__)
    def roiaware_pool3d_grad_cpu(self, pts_idx_of_voxels, argmax, grad_out,
                                        npoints, pool_method):
        channels = grad_out.shape[-1]
        grad_in = torch.zeros((npoints, channels)).type_as(grad_out)
        
        # cast
        dtype = grad_out.dtype
        if (dtype == torch.float16):
            grad_out_cast = grad_out.type(torch.float32)
            grad_in_cast = grad_in.type(torch.float32)
        else:
            grad_out_cast = grad_out
            grad_in_cast = grad_in

        # Compute
        if pool_method == 0:
            self.roiaware_maxpool3d_grad_cpu(argmax, grad_out_cast, grad_in_cast)
            
        elif pool_method == 1:
            self.roiaware_avgpool3d_grad_cpu(pts_idx_of_voxels, grad_out_cast, grad_in_cast)

        # cast
        if (dtype == torch.float16):
            grad_out_cast = grad_out_cast.type(torch.float16)
            grad_in_cast = grad_in_cast.type(torch.float16)
        else:
            grad_out_cast = grad_out
            grad_in_cast = grad_in
        
        return grad_in_cast

    def roiaware_pool3d_grad_npu(self, pts_idx_of_voxels, argmax, grad_out, npoints, pool_method):
        grad_in = mx_driving._C.roiaware_pool3d_grad(pts_idx_of_voxels, argmax, grad_out, npoints, pool_method)
        return grad_in
    
    def roiaware_maxpool3d_grad_cpu(self, argmax, grad_out, grad_in):
        boxes_num, out_x, out_y, out_z, channels = grad_out.shape
        npoints, _ = grad_in.shape

        for b in range(boxes_num):
            for ox in range(out_x):
                for oy in range(out_y):
                    for oz in range(out_z):
                        N_idx = argmax[b, ox, oy, oz, :]
                        C_idx = np.arange(channels)
                        grad_in[N_idx, C_idx] += grad_out[b, ox, oy, oz, C_idx]
    
    def roiaware_avgpool3d_grad_cpu(self, pts_idx_of_voxels, grad_out, grad_in):
        boxes_num, out_x, out_y, out_z, channels = grad_out.shape
        max_pts_per_voxel = pts_idx_of_voxels.shape[-1]

        for b in range(boxes_num):
            for ox in range(out_x):
                for oy in range(out_y):
                    for oz in range(out_z):
                        total_pts = pts_idx_of_voxels[b, ox, oy, oz, 0]
                        for i in range(1, total_pts + 1):
                            pts_idx = pts_idx_of_voxels[b, ox, oy, oz, i]
                            grad_in[pts_idx, :] += grad_out[b, ox, oy, oz, :] / max(total_pts, 1.0)

    @golden_data_cache(__file__)
    def gen_input_data(self, pts_idx_of_voxels_shape, channels, npoints, dtype):
        boxes_num, out_x, out_y, out_z, max_pts_per_voxel = pts_idx_of_voxels_shape
        grad_out = np.random.uniform(-5, 5, (boxes_num, out_x, out_y, out_z, channels)).astype(dtype)
        argmax = np.random.randint(0, npoints, (boxes_num, out_x, out_y, out_z, channels)).astype("int32")
        pts_idx_of_voxels = self.gen_pts_idx_of_voxels(pts_idx_of_voxels_shape, npoints).astype("int32")
        
        grad_out = torch.from_numpy(grad_out)
        argmax = torch.from_numpy(argmax)
        pts_idx_of_voxels = torch.from_numpy(pts_idx_of_voxels)
        return argmax, grad_out, pts_idx_of_voxels

    @golden_data_cache(__file__)
    def gen_pts_idx_of_voxels(self, pts_idx_of_voxels_shape, npoints):
        boxes_num, out_x, out_y, out_z, max_pts_per_voxel = pts_idx_of_voxels_shape
        pts_idx_of_voxels = np.zeros((boxes_num, out_x, out_y, out_z, max_pts_per_voxel - 1)).astype("int32")
        total_pts_array = np.random.randint(0, max_pts_per_voxel, (boxes_num, out_x, out_y, out_z))
        for b in range(boxes_num):
            for ox in range(out_x):
                for oy in range(out_y):
                    for oz in range(out_z):
                        total_pts = total_pts_array[b, ox, oy, oz]
                        choiced_idx = np.array(np.random.choice(npoints, total_pts, replace=False)).astype("int32")
                        choiced_idx = np.sort(choiced_idx)
                        pts_idx_of_voxels[b, ox, oy, oz, 0:total_pts] = choiced_idx
        pts_idx_of_voxels = np.concatenate([total_pts_array.reshape(boxes_num, out_x, out_y, out_z, 1), pts_idx_of_voxels], axis=-1)
        return pts_idx_of_voxels

    def one_case(self, boxes_num, out_size, channels, npoints, max_pts_per_voxel, pool_method, dtype):
        out_x, out_y, out_z = out_size
        pts_idx_of_voxels_shape = (boxes_num, out_x, out_y, out_z, max_pts_per_voxel)
        argmax, grad_out, pts_idx_of_voxels = self.gen_input_data(pts_idx_of_voxels_shape, channels, npoints, dtype)

        golden_grad_in = np.zeros((npoints, channels)).astype(dtype)
        golden_grad_in = torch.from_numpy(golden_grad_in)

        golden_grad_in = self.roiaware_pool3d_grad_cpu(pts_idx_of_voxels, argmax, grad_out, npoints, pool_method)

        grad_in = np.zeros((npoints, channels)).astype(dtype)
        grad_in = torch.from_numpy(grad_in)
        grad_out = grad_out.npu()
        argmax = argmax.npu()
        grad_in = grad_in.npu()
        pts_idx_of_voxels = pts_idx_of_voxels.npu()
        
        grad_in = self.roiaware_pool3d_grad_npu(pts_idx_of_voxels, argmax, grad_out, npoints, pool_method)
        
        self.assertRtolEqual(grad_in, golden_grad_in)
        
    def test_roiaware_pool3d_grad(self):
        out_size = (14, 14, 14)
        self.one_case(1, out_size, 256, 128, 128, 0, "float32")
        self.one_case(1, out_size, 256, 128, 128, 1, "float32")
        self.one_case(1, out_size, 256, 128, 128, 0, "float16")
        self.one_case(1, out_size, 256, 128, 128, 1, "float16")


if __name__ == "__main__":
    torch.npu.conv.allow_hf32 = False
    run_tests()