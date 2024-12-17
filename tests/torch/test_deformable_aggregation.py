import unittest

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving
import mx_driving.fused


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


@golden_data_cache(__file__)
def cpu_gen_inputs(B, C, anchor, pts, numGroups):
    feature_maps = np.random.rand(B, 2816, C).astype(np.float32)
    spatial_shape = torch.tensor([[[32, 88]]], dtype=torch.int32).numpy()
    scale_start_index = torch.tensor([[0]], dtype=torch.int32).numpy()
    sample_location = np.random.rand(B, anchor, pts, 1, 2).astype(np.float32)
    weights = np.random.rand(B, anchor, pts, 1, 1, numGroups).astype(np.float32)

    return feature_maps, spatial_shape, scale_start_index, sample_location, weights


class TestDeformableAggregation(TestCase):
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    @golden_data_cache(__file__)
    def golden_deformable_aggregation(self, batch_size, num_anchors, num_pts, num_cams, num_scale, num_embeds,
                                      num_groups, num_feat, feature_maps, spatial_shape, scale_start_index,
                                      sample_location, weights):

        out = np.zeros((batch_size, num_anchors, num_embeds)).astype(np.float32)

        num_kernels = batch_size * num_anchors * num_pts * num_cams * num_scale
        for idx in range(num_kernels):
            chanenl_offset = 0
            weights_offset = idx
            scale_index = idx % num_scale
            idx //= num_scale

            cam_index = idx % num_cams
            idx //= num_cams

            pts_index = idx % num_pts
            idx //= num_pts

            anchor_index = idx % num_anchors
            idx //= num_anchors

            batch_index = idx % batch_size
            idx //= batch_size

            loc_w = sample_location[batch_index, anchor_index, pts_index, cam_index, 0]
            loc_h = sample_location[batch_index, anchor_index, pts_index, cam_index, 1]

            if loc_w <= 0 or loc_w >= 1:
                continue
            if loc_h <= 0 or loc_h >= 1:
                continue

            scale_start_index_idx = scale_start_index[cam_index, scale_index]
            value_offset = (batch_index * num_feat + scale_start_index_idx) * num_embeds

            h = spatial_shape[cam_index, scale_index, 0]
            w = spatial_shape[cam_index, scale_index, 1]

            h_im = loc_h * h - 0.5
            w_im = loc_w * w - 0.5

            h_low = np.floor(h_im).astype(int)
            w_low = np.floor(w_im).astype(int)
            h_high = h_low + 1
            w_high = w_low + 1
            lh = h_im - h_low
            lw = w_im - w_low
            hh = 1 - lh
            hw = 1 - lw

            w_stride = num_embeds
            h_stride = w * w_stride

            h_low_ptr_offset = h_low * h_stride
            h_high_ptr_offset = h_low_ptr_offset + h_stride

            w_low_ptr_offset = w_low * w_stride
            w_high_ptr_offset = w_low_ptr_offset + w_stride
            for groups_idx in range(num_groups):

                weights_idx = weights_offset * num_groups + groups_idx % num_groups
                weight = weights[weights_idx]

                v1 = 0
                if h_low >= 0 and w_low >= 0:
                    ptr1 = value_offset + h_low_ptr_offset + w_low_ptr_offset + chanenl_offset
                    v1 = feature_maps[ptr1 : ptr1 + num_embeds // num_groups]

                v2 = 0
                if h_low >= 0 and w_high <= w - 1:
                    ptr2 = value_offset + h_low_ptr_offset + w_high_ptr_offset + chanenl_offset
                    v2 = feature_maps[ptr2 : ptr2 + num_embeds // num_groups]

                v3 = 0
                if h_high <= h - 1 and w_low >= 0:
                    ptr3 = value_offset + h_high_ptr_offset + w_low_ptr_offset + chanenl_offset
                    v3 = feature_maps[ptr3 : ptr3 + num_embeds // num_groups]

                v4 = 0
                if h_high <= h - 1 and w_high <= w - 1:
                    ptr4 = value_offset + h_high_ptr_offset + w_high_ptr_offset + chanenl_offset
                    v4 = feature_maps[ptr4 : ptr4 + num_embeds // num_groups]

                w1 = hh * hw
                w2 = hh * lw
                w3 = lh * hw
                w4 = lh * lw

                val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4) * weight

                out[batch_index, anchor_index, chanenl_offset : chanenl_offset + num_embeds // num_groups] += val

                chanenl_offset += num_embeds // num_groups

        return out


    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `DeformableAggregation` is only supported on 910B, skip this ut!")
    def test_deformable_aggregation(self):
        np.random.seed(50051)

        bList = [1, 5, 10]
        cList = [32, 64]
        numGroupsList = [8, 16]
        anchorList = [10, 13, 18]
        ptsList = [10, 50, 31]
        for B in bList:
            for C in cList:
                for pts in ptsList:
                    for anchor in anchorList:
                        for numGroups in numGroupsList:
                            feature_maps, spatial_shape, scale_start_index, sample_location, weights = cpu_gen_inputs(B, C, anchor, pts, numGroups)

                            torch_feature_maps = torch.from_numpy(feature_maps).npu()
                            torch_spatial_shape = torch.from_numpy(spatial_shape).npu()
                            torch_scale_start_index = torch.from_numpy(scale_start_index).npu()
                            torch_sample_location = torch.from_numpy(sample_location).npu()
                            torch_weights = torch.from_numpy(weights).npu()

                            batch_size = feature_maps.shape[0]
                            num_feat = feature_maps.shape[1]
                            num_embeds = feature_maps.shape[2]
                            num_cams = spatial_shape.shape[0]
                            num_scale = spatial_shape.shape[1]
                            num_anchors = sample_location.shape[1]
                            num_pts = sample_location.shape[2]
                            num_groups = weights.shape[5]

                            weights = weights.flatten()
                            feature_maps = feature_maps.flatten()


                            out_cpu = self.golden_deformable_aggregation(batch_size, num_anchors, num_pts, num_cams,
                                                               num_scale, num_embeds, num_groups, num_feat,
                                                               feature_maps, spatial_shape, scale_start_index,
                                                               sample_location, weights)
                            out_npu = mx_driving.fused.npu_deformable_aggregation(torch_feature_maps,
                                                                                   torch_spatial_shape,
                                                                                   torch_scale_start_index,
                                                                                   torch_sample_location,
                                                                                   torch_weights)

                            self.assertRtolEqual(out_cpu, out_npu.cpu().numpy())

                            out_npu_new = mx_driving.deformable_aggregation(torch_feature_maps,
                                                                            torch_spatial_shape,
                                                                            torch_scale_start_index,
                                                                            torch_sample_location,
                                                                            torch_weights)

                            self.assertRtolEqual(out_cpu, out_npu_new.cpu().numpy())


if __name__ == "__main__":
    run_tests()
