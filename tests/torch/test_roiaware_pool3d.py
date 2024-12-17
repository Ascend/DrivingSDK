import math
import unittest

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving
import mx_driving.detection


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestRoIAwarePool3dGrad(TestCase):
    @golden_data_cache(__file__)
    def roiaware_pool3d_cpu(self, rois, pts, pts_feature, out, max_pts_per_voxel, pool_method, dtype):
        # cast
        if (dtype == np.float16):
            rois_cast = rois.astype(np.float32)
            pts_cast = pts.astype(np.float32)
            pts_feature_cast = pts_feature.astype(np.float32)
        elif(dtype == np.float32):
            rois_cast = rois
            pts_cast = pts
            pts_feature_cast = pts_feature

        # Compute
        pooled_features_cpu = self.roiaware_pool3d_golden(rois_cast, pts_cast, pts_feature_cast, out, max_pts_per_voxel, pool_method)

        # cast
        if (dtype == np.float16):
            pooled_features_cpu_cast = pooled_features_cpu.astype(np.float16)
        else:
            pooled_features_cpu_cast = pooled_features_cpu.astype(np.float32)
        
        return pooled_features_cpu_cast

    def roiaware_pool3d_npu(self, rois, pts, pts_feature, out_size, mode, max_pts_per_voxel, dtype):
        
        rois_input = torch.tensor(rois).npu()
        pts_input = torch.tensor(pts).npu()
        pts_feature_input = torch.tensor(pts_feature).npu()
        
        pool_mapping = {'max': 0, 'avg': 1}
        pool_method = pool_mapping[mode]
            
        num_rois = rois.shape[0]
        num_channels = pts_feature.shape[-1]
        num_pts = pts.shape[0]
        out_x = out_size[0]
        out_y = out_size[1]
        out_z = out_size[2]
        
        pooled_features_npu_old = mx_driving.detection.roiaware_pool3d(rois_input, pts_input, pts_feature_input, out_size, max_pts_per_voxel, pool_method)
        pooled_features_npu = mx_driving.roiaware_pool3d(rois_input, pts_input, pts_feature_input, out_size, max_pts_per_voxel, pool_method)
        
        return pooled_features_npu, pooled_features_npu_old
    
    def lidar_to_local_coords(self, shift_x, shift_y, rz):
        cosa = math.cos(-rz)
        sina = math.sin(-rz)
        local_x = shift_x * cosa + shift_y * (-sina)
        local_y = shift_x * sina + shift_y * cosa
        
        return local_x, local_y
    
    def check_pt_in_box3d(self, pt, box3d):
        x = pt[0]
        y = pt[1]
        z = pt[2]
        
        cx = box3d[0]
        cy = box3d[1]
        cz = box3d[2]
        x_size = box3d[3]
        y_size = box3d[4]
        z_size = box3d[5]
        rz = box3d[6]
        
        cz += z_size / 2.0
        
        if (abs(z - cz) > z_size / 2.0):
            return 0, 0, 0
        
        local_x, local_y = self.lidar_to_local_coords(x - cx, y - cy, rz) 
        if (local_x > -x_size / 2.0) and (local_x < x_size / 2.0) and (local_y > -y_size / 2.0) and (local_y < y_size / 2.0):
            return 1, local_x, local_y
        else:
            return 0, local_x, local_y
    
    @golden_data_cache(__file__)
    def roiaware_pool3d_golden(self, rois, pts, pts_feature, out, max_pts_per_voxel, mode):
        num_rois = rois.shape[0]
        num_channels = pts_feature.shape[-1]
        num_pts = pts.shape[0]
        
        pooled_features = np.zeros((num_rois, out[0], out[1], out[2], num_channels))
        argmax = np.zeros(shape=(num_rois, out[0], out[1], out[2], num_channels), dtype=int)
        pts_idx_of_voxels = np.zeros(shape=(num_rois, out[0], out[1], out[2], max_pts_per_voxel), dtype=int)
        
        pts_mask = np.ones(shape=(num_rois, num_pts), dtype=int)
        for i in range(num_pts):
            for j in range(num_rois):
                cur_in_flag, local_x, local_y = self.check_pt_in_box3d(pts[i, :], rois[j, :])
                pts_mask[j, i] = -1
                if(cur_in_flag > 0):
                    local_z = pts[i, 2] - rois[j, 2]
                    x_size = rois[j, 3]
                    y_size = rois[j, 4]
                    z_size = rois[j, 5]
                    
                    x_res = x_size / out[0]
                    y_res = y_size / out[1]
                    z_res = z_size / out[2]
                    
                    x_idx = int((local_x + x_size / 2) / x_res)
                    y_idx = int((local_y + y_size / 2) / y_res)
                    z_idx = int(local_z / z_res)
                    
                    indx_encoding = (x_idx << 16) + (y_idx << 8) + z_idx
                    pts_mask[j, i] = indx_encoding
        
        decoder = 0xFF
        for i in range(num_rois):
            for j in range(num_pts):
                max_num_pts = max_pts_per_voxel - 1
                if(pts_mask[i, j] != -1):
                    idx_encoding = pts_mask[i, j]
                    x_idx = (idx_encoding >> 16) & decoder
                    y_idx = (idx_encoding >> 8) & decoder
                    z_idx = idx_encoding & decoder
                    
                    x_idx = min(max(x_idx, 0), out[0] - 1)
                    y_idx = min(max(y_idx, 0), out[1] - 1)
                    z_idx = min(max(z_idx, 0), out[2] - 1)
                
                    cnt = pts_idx_of_voxels[i, x_idx, y_idx, z_idx, 0]
                    if(cnt < max_num_pts):
                        pts_idx_of_voxels[i, x_idx, y_idx, z_idx, 0 + cnt + 1] = j
                        pts_idx_of_voxels[i, x_idx, y_idx, z_idx, 0] += 1

        if(mode == 'max'):
            for i in range(out[0]):
                for j in range(out[1]):
                    for k in range(out[2]):
                        for box_idx in range(num_rois):
                            for c_idx in range(num_channels):
                                argmax_idx = -1
                                max_val = -1e10
                                total_pts = pts_idx_of_voxels[box_idx, i, j, k, 0]
                                for p_idx in range(1, total_pts + 1):
                                    if(pts_feature[pts_idx_of_voxels[box_idx, i, j, k, p_idx], c_idx] > max_val):
                                        max_val = pts_feature[pts_idx_of_voxels[box_idx, i, j, k, p_idx], c_idx]
                                        argmax_idx = pts_idx_of_voxels[box_idx, i, j, k, p_idx]
                                        
                                if(argmax_idx != -1):
                                    pooled_features[box_idx, i, j, k, c_idx] = max_val
                                argmax[box_idx, i, j, k, c_idx] = argmax_idx    
        elif(mode == 'avg'):
            for i in range(out[0]):
                for j in range(out[1]):
                    for k in range(out[2]):
                        for box_idx in range(num_rois):
                            for c_idx in range(num_channels):
                                sum_val = 0
                                total_pts = pts_idx_of_voxels[box_idx, i, j, k, 0]
                                for p_idx in range(1, total_pts + 1):
                                    sum_val += pts_feature[pts_idx_of_voxels[box_idx, i, j, k, p_idx], c_idx]
                                    
                                if(total_pts > 0):
                                    pooled_features[box_idx, i, j, k, c_idx] = sum_val / total_pts
        
        return pooled_features
    
    @golden_data_cache(__file__)
    def gen_input_data(self, boxes_num, out_size, channels, npoints, dtype):
        xyz_coor = np.random.uniform(-1, 1, size=(boxes_num, 3)).astype(dtype)
        xyz_size_num = np.random.uniform(1, 50, size=(1, 3)).astype(dtype)
        xyz_size = (xyz_size_num * np.ones((boxes_num, 3))).astype(dtype)
        angle = np.radians(np.random.randint(0, 360, size=(boxes_num, 1))).astype(dtype)

        rois = np.concatenate((xyz_coor, xyz_size), axis=1)
        rois = np.concatenate((rois, angle), axis=1)

        pts = np.random.uniform(-2, 4, size=(npoints, 3)).astype(dtype)
        pts_feature = np.random.uniform(-1, 1, size=(npoints, channels)).astype(dtype)
        
        return rois, pts, pts_feature


    def one_case(self, boxes_num, out_size, channels, npoints, max_pts_per_voxel, pool_method, dtype):
        rois, pts, pts_feature = self.gen_input_data(boxes_num, out_size, channels, npoints, dtype)

        pooled_features_cpu = self.roiaware_pool3d_cpu(rois, pts, pts_feature, out_size, max_pts_per_voxel, pool_method, dtype)
        pooled_features_npu, pooled_features_npu_old = self.roiaware_pool3d_npu(rois, pts, pts_feature, out_size, pool_method, max_pts_per_voxel, dtype)
        pooled_features_cpu_tensor = torch.tensor(pooled_features_cpu)
        
        self.assertRtolEqual(pooled_features_cpu_tensor, pooled_features_npu.detach().cpu())
        self.assertRtolEqual(pooled_features_cpu_tensor, pooled_features_npu_old.detach().cpu())
        
    def test_roiaware_pool3d(self):
        out_size = (4, 4, 4)
        self.one_case(10, out_size, 256, 128, 128, 'max', np.float32)
        self.one_case(10, out_size, 256, 128, 128, 'avg', np.float32)
        self.one_case(20, out_size, 512, 256, 128, 'max', np.float16)
        self.one_case(20, out_size, 512, 256, 128, 'avg', np.float16)


if __name__ == "__main__":
    torch.npu.conv.allow_hf32 = False
    run_tests()