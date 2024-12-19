import unittest

import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

from mx_driving import npu_gaussian


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def gaussian_radius(det_size, min_overlap=0.5):
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


# pylint: disable=too-many-arguments,huawei-too-many-arguments
@golden_data_cache(__file__)
def golden_gaussian(
    boxes,
    out_size_factor,
    gaussian_overlap,
    min_radius,
    voxel_size_x,
    voxel_size_y,
    pc_range_x,
    pc_range_y,
    feature_map_size_x,
    feature_map_size_y,
    norm_bbox,
    with_velocity,
):
    max_objs = 500
    if with_velocity:
        anno_box = torch.zeros([max_objs, 10], dtype=torch.float32)
    else:
        anno_box = torch.zeros([max_objs, 8], dtype=torch.float32)

    ind = torch.zeros(max_objs, dtype=torch.int64)
    mask = torch.zeros(max_objs, dtype=torch.uint8)

    for k in range(boxes.size(0)):
        width = boxes[k][3]
        length = boxes[k][4]
        width = width / voxel_size_x / out_size_factor
        length = length / voxel_size_y / out_size_factor

        if width > 0 and length > 0:
            radius = gaussian_radius((length, width), min_overlap=gaussian_overlap)
            radius = max(min_radius, int(radius))

            x, y, z = boxes[k][0], boxes[k][1], boxes[k][2]

            coor_x = (x - pc_range_x) / voxel_size_x / out_size_factor
            coor_y = (y - pc_range_y) / voxel_size_y / out_size_factor
            center = torch.tensor([coor_x, coor_y], dtype=torch.float32)
            center_int = center.to(torch.int32)

            if not (
                0 <= center_int[0] < feature_map_size_x
                and 0 <= center_int[1] < feature_map_size_y
            ):
                continue

            new_idx = k
            x, y = center_int[0], center_int[1]

            assert y * feature_map_size_x + x < feature_map_size_x * feature_map_size_y

            ind[new_idx] = y * feature_map_size_x + x
            mask[new_idx] = 1
            rot = boxes[k][6]
            box_dim = boxes[k][3:6]
            if norm_bbox:
                box_dim = box_dim.log()
            if with_velocity:
                vx, vy = boxes[k][7:]
                anno_box[new_idx] = torch.cat(
                    [
                        center - torch.tensor([x, y]),
                        z.unsqueeze(0),
                        box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0),
                        vx.unsqueeze(0),
                        vy.unsqueeze(0),
                    ]
                )
            else:
                anno_box[new_idx] = torch.cat(
                    [
                        center - torch.tensor([x, y]),
                        z.unsqueeze(0),
                        box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0),
                    ]
                )
    return mask, ind, anno_box


class TestGaussian(TestCase):
    seed = 1024
    torch.manual_seed(seed)

    def test_npu_gaussian(self):
        shapes = [
            [31, 9],
            [100, 9],
            [251, 9],
            [500, 9],
        ]
        out_size_factor = 8
        gaussian_overlap = 0.1
        min_radius = 2
        voxel_size_x = 0.1
        voxel_size_y = 0.1
        pc_range_x = -51.2
        pc_range_y = -51.2
        feature_map_size_x = 128
        feature_map_size_y = 128
        norm_bbox = True
        with_velocity = True
        for shape in shapes:
            H, W = shape
            boxes = 50 * torch.rand((H, W), dtype=torch.float32)
            boxes_npu = boxes.clone().to("npu")

            mask_cpu, ind_cpu, anno_box_cpu = golden_gaussian(
                boxes,
                out_size_factor,
                gaussian_overlap,
                min_radius,
                voxel_size_x,
                voxel_size_y,
                pc_range_x,
                pc_range_y,
                feature_map_size_x,
                feature_map_size_y,
                norm_bbox,
                with_velocity,
            )
            output_npu = npu_gaussian(
                boxes_npu,
                out_size_factor,
                gaussian_overlap,
                min_radius,
                voxel_size_x,
                voxel_size_y,
                pc_range_x,
                pc_range_y,
                feature_map_size_x,
                feature_map_size_y,
                norm_bbox,
                with_velocity,
            )
            mask_npu = output_npu[2]
            ind_npu = output_npu[3]
            anno_box_npu = output_npu[4]

            self.assertRtolEqual(mask_cpu.numpy(), mask_npu.cpu().numpy())
            self.assertRtolEqual(ind_cpu.numpy(), ind_npu.cpu().numpy())
            self.assertRtolEqual(anno_box_cpu.numpy(), anno_box_npu.cpu().numpy())


if __name__ == "__main__":
    run_tests()
