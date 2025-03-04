import unittest

import torch
import torch_npu
import numpy as np

from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

from mx_driving import npu_assign_target_of_single_head


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


# pylint: disable=too-many-arguments,huawei-too-many-arguments
@golden_data_cache(__file__)
def golden_assign_target_of_single_head(
    boxes,
    cur_class_id_tensor,
    num_classes,
    out_size_factor,
    gaussian_overlap,
    min_radius,
    voxel_size,
    pc_range,
    feature_map_size,
    norm_bbox,
    with_velocity,
    flip_angle,
    max_objs=500
):
    heatmap = boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
    ret_boxes = boxes.new_zeros((max_objs, boxes.shape[-1] + 1))
    inds = boxes.new_zeros(max_objs).long()
    mask = boxes.new_zeros(max_objs, dtype=torch.uint8)
    x, y, z = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    coord_x = (x - pc_range[0]) / voxel_size[0] / out_size_factor
    coord_y = (y - pc_range[1]) / voxel_size[1] / out_size_factor
    coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)
    coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)
    center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
    center_int = center.int()
    center_int_float = center_int.float()

    dx, dy, dz = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    dx = dx / voxel_size[0] / out_size_factor
    dy = dy / voxel_size[1] / out_size_factor

    radius = gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
    radius = torch.clamp_min(radius.int(), min=min_radius)
    for k in range(min(max_objs, boxes.shape[0])):
        if dx[k] <= 0 or dy[k] <= 0:
            continue
        if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
            continue
        cur_class_id = (cur_class_id_tensor[k] - 1).long()
        draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

        inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
        mask[k] = 1
        ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
        ret_boxes[k, 2] = z[k]
        if norm_bbox == True:
            ret_boxes[k, 3:6] = boxes[k, 3:6].log()
        else:
            ret_boxes[k, 3:6] = boxes[k, 3:6]
        if flip_angle == True:
            ret_boxes[k, 6] = torch.cos(boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(boxes[k, 6])
        else:
            ret_boxes[k, 6] = torch.sin(boxes[k, 6])
            ret_boxes[k, 7] = torch.cos(boxes[k, 6])
        if boxes.shape[1] > 8:
            ret_boxes[k, 8:] = boxes[k, 7:]

    return heatmap, ret_boxes, inds, mask


def gaussian_radius(height, width, min_overlap=0.5):
    """
    Args:
        height: (N)
        width: (N)
        min_overlap:
    Returns:
    """
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1 ** 2 - 4 * a1 * c1).sqrt()
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = (b2 ** 2 - 4 * a2 * c2).sqrt()
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
    r3 = (b3 + sq3) / 2
    ret = torch.min(torch.min(r1, r2), r3)
    return ret


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian_to_heatmap(heatmap, center, radius, k=1, valid_mask=None):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom, radius - left:radius + right]
    ).to(heatmap.device).float()

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        if valid_mask is not None:
            cur_valid_mask = valid_mask[y - top:y + bottom, x - left:x + right]
            masked_gaussian = masked_gaussian * cur_valid_mask.float()

        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


class TestAssignTargetOfSingleHead(TestCase):
    seed = 1024
    torch.manual_seed(seed)

    def test_npu_assign_target_of_single_head(self):
        shapes = [
            [31, 9],
            [100, 9],
            [251, 9],
            [500, 9],
        ]
        out_size_factor = 8
        gaussian_overlap = 0.1
        min_radius = 2
        voxel_size = [0.1, 0.1]
        pc_range = [-51.2, -51.2]
        feature_map_size = [128, 128]
        num_classes = 1
        for shape in shapes:
            H, W = shape
            boxes = 50 * torch.rand((H, W), dtype=torch.float32)
            cur_class_id = torch.ones((H), dtype=torch.int32)
            boxes_npu = boxes.clone().to("npu")
            cur_class_id_npu = cur_class_id.to("npu")

            heatmap_cpu, anno_box_cpu, ind_cpu, mask_cpu = golden_assign_target_of_single_head(
                boxes, cur_class_id, num_classes, out_size_factor, gaussian_overlap, min_radius, voxel_size, pc_range, feature_map_size, True, True, True
            )
            heatmap_npu, ann_box_npu, ind_npu, mask_npu = npu_assign_target_of_single_head(
                boxes_npu, cur_class_id_npu, num_classes, out_size_factor, gaussian_overlap, min_radius, voxel_size, pc_range, feature_map_size, True, True, True
            )
            self.assertRtolEqual(heatmap_cpu.numpy(), heatmap_npu.cpu().numpy())
            self.assertRtolEqual(anno_box_cpu.numpy(), ann_box_npu.cpu().numpy())
            self.assertRtolEqual(ind_cpu.numpy(), ind_npu.cpu().numpy())
            self.assertRtolEqual(mask_cpu.numpy(), mask_npu.cpu().numpy())


if __name__ == "__main__":
    run_tests()
