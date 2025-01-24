import unittest

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving
import mx_driving.fused


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


# 'pylint: disable=too-many-arguments,huawei-too-many-arguments
@golden_data_cache(__file__)
def gen_inputs(B, C, input_h, input_w, anchor, pts, numGroups):
    feature_maps = np.random.rand(B, input_h * input_w, C).astype(np.float32)
    spatial_shape = torch.tensor([[[input_h, input_w]]], dtype=torch.int32).numpy()
    scale_start_index = torch.tensor([[0]], dtype=torch.int32).numpy()
    sample_location = np.random.rand(B, anchor, pts, 1, 2).astype(np.float32)
    weights = np.random.rand(B, anchor, pts, 1, 1, numGroups).astype(np.float32)

    return feature_maps, spatial_shape, scale_start_index, sample_location, weights


class TestDeformableAggregation(TestCase):
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    @golden_data_cache(__file__)
    def golden_deformable_aggregation_grad(
        self,
        batch_size,
        num_anchors,
        num_pts,
        num_cams,
        num_scale,
        num_embeds,
        num_groups,
        num_feat,
        feature_maps,
        spatial_shape,
        scale_start_index,
        sample_location,
        weights
    ):

        out_cpu = np.zeros((batch_size, num_anchors, num_embeds)).astype(np.float32)
        grad_mc_ms_feat = np.zeros_like(feature_maps)
        grad_sampling_location = np.zeros_like(sample_location)
        grad_weights = np.zeros_like(weights)
        grad_output = np.ones_like(out_cpu)

        feature_maps = feature_maps.flatten()
        spatial_shape = spatial_shape.flatten()
        scale_start_index = scale_start_index.flatten()
        sample_location = sample_location.flatten()
        weights = weights.flatten()
        grad_mc_ms_feat = grad_mc_ms_feat.flatten()
        grad_sampling_location = grad_sampling_location.flatten()
        grad_weights = grad_weights.flatten()
        grad_output = grad_output.flatten()

        num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale
        for idx in range(num_kernels):

            weights_ptr = idx // (num_embeds // num_groups)
            channel_index = idx % num_embeds
            idx //= num_embeds

            scale_index = idx % num_scale
            idx //= num_scale

            cam_index = idx % num_cams
            idx //= num_cams

            pts_index = idx % num_pts
            idx //= num_pts

            anchor_index = idx % num_anchors
            idx //= num_anchors

            batch_index = idx % batch_size

            anchor_index = batch_index * num_anchors + anchor_index
            loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1

            loc_w = sample_location[loc_offset]
            if loc_w <= 0 or loc_w >= 1:
                continue
            loc_h = sample_location[loc_offset + 1]
            if loc_h <= 0 or loc_h >= 1:
                continue

            grad = grad_output[anchor_index * num_embeds + channel_index]

            cam_scale_index = cam_index * num_scale + scale_index
            value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index

            cam_scale_index = cam_scale_index << 1

            h = spatial_shape[cam_scale_index]
            w = spatial_shape[cam_scale_index + 1]

            h_im = loc_h * h - 0.5
            w_im = loc_w * w - 0.5

            weight = weights[weights_ptr]

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

            w1 = hh * hw
            w2 = hh * lw
            w3 = lh * hw
            w4 = lh * lw

            top_grad_mc_ms_feat = grad * weight

            grad_h_weight = 0
            grad_w_weight = 0

            v1 = 0
            if h_low >= 0 and w_low >= 0:
                ptr1 = value_offset + h_low_ptr_offset + w_low_ptr_offset
                v1 = feature_maps[ptr1]
                grad_h_weight -= hw * v1
                grad_w_weight -= hh * v1
                grad_mc_ms_feat[ptr1] += w1 * top_grad_mc_ms_feat

            v2 = 0
            if h_low >= 0 and w_high <= w - 1:
                ptr2 = value_offset + h_low_ptr_offset + w_high_ptr_offset
                v2 = feature_maps[ptr2]
                grad_h_weight -= lw * v2
                grad_w_weight += hh * v2
                grad_mc_ms_feat[ptr2] += w2 * top_grad_mc_ms_feat

            v3 = 0
            if h_high <= h - 1 and w_low >= 0:
                ptr3 = value_offset + h_high_ptr_offset + w_low_ptr_offset
                v3 = feature_maps[ptr3]
                grad_h_weight += hw * v3
                grad_w_weight -= lh * v3
                grad_mc_ms_feat[ptr3] += w3 * top_grad_mc_ms_feat

            v4 = 0
            if h_high <= h - 1 and w_high <= w - 1:
                ptr4 = value_offset + h_high_ptr_offset + w_high_ptr_offset
                v4 = feature_maps[ptr4]
                grad_h_weight += lw * v4
                grad_w_weight += lh * v4
                grad_mc_ms_feat[ptr4] += w4 * top_grad_mc_ms_feat

            val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4)

            grad_weights[weights_ptr] += grad * val

            grad_sampling_location[loc_offset] += w * grad_w_weight * top_grad_mc_ms_feat
            grad_sampling_location[loc_offset + 1] += h * grad_h_weight * top_grad_mc_ms_feat

        return grad_mc_ms_feat, grad_sampling_location, grad_weights

    @unittest.skipIf(
        DEVICE_NAME != 'Ascend910B',
        "OP `DeformableAggregationGrad` is only supported on 910B, skip this ut!",
    )
    def test_deformable_aggregation(self):
        np.random.seed(50051)

        bList = [1, 5]
        cList = [8 * 8, 8 * 8 * 2]
        ptsList = [10, 21]
        anchorList = [10, 13]
        numGroupsList = [8]

        for B in bList:
            for C in cList:
                for pts in ptsList:
                    for anchor in anchorList:
                        for numGroups in numGroupsList:
                            input_h = 16
                            input_w = 22

                            feature_maps, spatial_shape, scale_start_index, sample_location, weights = gen_inputs(B, C, input_h, input_w, anchor, pts, numGroups)
                            feature_maps_shape = feature_maps.shape

                            torch_feature_maps = torch.from_numpy(feature_maps).npu()
                            torch_feature_maps.requires_grad = True
                            torch_spatial_shape = torch.from_numpy(spatial_shape).npu()
                            torch_scale_start_index = torch.from_numpy(scale_start_index).npu()
                            torch_sample_location = torch.from_numpy(sample_location).npu()
                            torch_sample_location.requires_grad = True
                            torch_weights = torch.from_numpy(weights).npu()
                            torch_weights.requires_grad = True

                            torch_feature_maps_new = torch.from_numpy(feature_maps).npu()
                            torch_feature_maps_new.requires_grad = True
                            torch_spatial_shape_new = torch.from_numpy(spatial_shape).npu()
                            torch_scale_start_index_new = torch.from_numpy(scale_start_index).npu()
                            torch_sample_location_new = torch.from_numpy(sample_location).npu()
                            torch_sample_location_new.requires_grad = True
                            torch_weights_new = torch.from_numpy(weights).npu()
                            torch_weights_new.requires_grad = True

                            batch_size = feature_maps.shape[0]
                            num_feat = feature_maps.shape[1]
                            num_embeds = feature_maps.shape[2]
                            num_cams = spatial_shape.shape[0]
                            num_scale = spatial_shape.shape[1]
                            num_anchors = sample_location.shape[1]
                            num_pts = sample_location.shape[2]
                            num_groups = weights.shape[5]


                            grad_mc_ms_feat, grad_sampling_location, grad_weights = self.golden_deformable_aggregation_grad(
                                batch_size,
                                num_anchors,
                                num_pts,
                                num_cams,
                                num_scale,
                                num_embeds,
                                num_groups,
                                num_feat,
                                feature_maps,
                                spatial_shape,
                                scale_start_index,
                                sample_location,
                                weights
                            )


                            out_npu = mx_driving.fused.npu_deformable_aggregation(
                                torch_feature_maps,
                                torch_spatial_shape,
                                torch_scale_start_index,
                                torch_sample_location,
                                torch_weights,
                            )

                            out_npu.backward(torch.ones_like(out_npu))

                            torch_grad_mc_ms_feat = torch_feature_maps.grad
                            torch_grad_sampling_location = torch_sample_location.grad
                            torch_grad_weights = torch_weights.grad

                            grad_mc_ms_feat = grad_mc_ms_feat.reshape(feature_maps_shape)
                            grad_sampling_location = grad_sampling_location.reshape(B, anchor, pts, 1, 2)
                            grad_weights = grad_weights.reshape(B, anchor, pts, 1, 1, numGroups)

                            torch_grad_mc_ms_feat = torch_grad_mc_ms_feat.cpu().numpy()
                            torch_grad_sampling_location = torch_grad_sampling_location.cpu().numpy()
                            torch_grad_weights = torch_grad_weights.cpu().numpy()

                            self.assertRtolEqual(grad_mc_ms_feat, torch_grad_mc_ms_feat, prec=0.00048828125)
                            self.assertRtolEqual(
                                grad_sampling_location,
                                torch_grad_sampling_location,
                                prec=0.00048828125,
                            )
                            self.assertRtolEqual(grad_weights, torch_grad_weights, prec=0.00048828125)

                            out_npu_new = mx_driving.deformable_aggregation(
                                torch_feature_maps_new,
                                torch_spatial_shape_new,
                                torch_scale_start_index_new,
                                torch_sample_location_new,
                                torch_weights_new,
                            )

                            out_npu_new.backward(torch.ones_like(out_npu_new))

                            torch_grad_mc_ms_feat_new = torch_feature_maps_new.grad
                            torch_grad_sampling_location_new = torch_sample_location_new.grad
                            torch_grad_weights_new = torch_weights_new.grad

                            torch_grad_mc_ms_feat_new = torch_grad_mc_ms_feat_new.cpu().numpy()
                            torch_grad_sampling_location_new = torch_grad_sampling_location_new.cpu().numpy()
                            torch_grad_weights_new = torch_grad_weights_new.cpu().numpy()

                            self.assertRtolEqual(grad_mc_ms_feat, torch_grad_mc_ms_feat_new, prec=0.00048828125)
                            self.assertRtolEqual(
                                grad_sampling_location,
                                torch_grad_sampling_location_new,
                                prec=0.00048828125,
                            )
                            self.assertRtolEqual(grad_weights, torch_grad_weights_new, prec=0.00048828125)


if __name__ == "__main__":
    run_tests()
