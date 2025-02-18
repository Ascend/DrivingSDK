"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2025-01-06
Modification Description:
Modification 1. Add support for Ascend NPU
"""

from typing import Tuple, Union
import torch
from torch.autograd import Function
from torch.nn import Module
from torch import Tensor
import torch_npu
import mx_driving._C

EPSILON = 1e-6
INF = torch.inf


class _DiffIouRotatedSortVertices(Function):
    @staticmethod
    def forward(ctx, vertices, mask, num_valid):
        sortedIdx = mx_driving._C.diff_iou_rotated_sort_vertices(vertices, mask, num_valid)
        return sortedIdx
    
    @staticmethod
    def backward(ctx, gradout):
        return ()


class DiffIouRotated(Module):
    def __init__(self):
        super(DiffIouRotated, self).__init__()
    
    def _box_intersection(self, corners1: Tensor,
                        corners2: Tensor) -> Tuple[Tensor, Tensor]:
        line1 = torch.cat([corners1, corners1[:, :, [1, 2, 3, 0], :]], dim=3)
        line2 = torch.cat([corners2, corners2[:, :, [1, 2, 3, 0], :]], dim=3)
        line1_ext = line1.unsqueeze(3)
        line2_ext = line2.unsqueeze(2)
        x1, y1, x2, y2 = line1_ext.split([1, 1, 1, 1], dim=-1)
        x3, y3, x4, y4 = line2_ext.split([1, 1, 1, 1], dim=-1)
        numerator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        denumerator_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        t = denumerator_t / (numerator)
        mask_t = (t >= 0) & (t <= 1)
        denumerator_u = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
        u = -denumerator_u / (numerator)
        mask_u = (u >= 0) & (u <= 1)
        mask = mask_t * mask_u
        t = denumerator_t / (numerator + EPSILON)
        intersections = torch.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)],
                                    dim=-1)
        intersections = intersections * mask.float().unsqueeze(-1)
        return intersections, mask

    def _box1_in_box2(self, corners1: Tensor, corners2: Tensor) -> Tensor:
        corners_related = corners2[:, :, None, :, :] - corners1[:, :, :, None, :]
        x = corners_related[..., 0]
        y = corners_related[..., 1]
        line_to_x = x[..., [1, 2, 3, 0]]
        line_to_y = y[..., [1, 2, 3, 0]]
        mask1 = (line_to_y > 0) ^ (y > 0)
        mask2 = x - y * (line_to_x - x) / (line_to_y - y) > 0
        return (mask1 & mask2).sum(-1) % 2 == 1

    def _box_in_box(self, corners1: Tensor, corners2: Tensor) -> Tuple[Tensor, Tensor]:
        c1_in_2 = self._box1_in_box2(corners1, corners2)
        c2_in_1 = self._box1_in_box2(corners2, corners1)
        return c1_in_2, c2_in_1

    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def _build_vertices(self, corners1: Tensor, corners2: Tensor, c1_in_2: Tensor,
                    c2_in_1: Tensor, intersections: Tensor,
                    valid_mask: Tensor) -> Tuple[Tensor, Tensor]:
        B = corners1.size()[0]
        N = corners1.size()[1]
        vertices = torch.cat(
            [corners1, corners2,
            intersections.view([B, N, -1, 2])], dim=2)
        mask = torch.cat([c1_in_2, c2_in_1, valid_mask.view([B, N, -1])], dim=2)
        return vertices, mask

    def _sort_indices(self, vertices: Tensor, mask: Tensor) -> Tensor:
        num_valid = torch.sum(mask.int(), dim=2).int()
        mean = torch.sum(
            vertices * mask.float().unsqueeze(-1), dim=2,
            keepdim=True) / num_valid.unsqueeze(-1).unsqueeze(-1)
        vertices_normalized = vertices - mean
        return _DiffIouRotatedSortVertices.apply(vertices_normalized, mask, num_valid).long()

    def _calculate_area(self, idx_sorted: Tensor,
                    vertices: Tensor) -> Tuple[Tensor, Tensor]:
        idx_ext = idx_sorted.unsqueeze(-1).repeat([1, 1, 1, 2])
        selected = torch.gather(vertices, 2, idx_ext)
        total = selected[:, :, 0:-1, 0] * selected[:, :, 1:, 1] \
            - selected[:, :, 0:-1, 1] * selected[:, :, 1:, 0]
        total = torch.sum(total, dim=2)
        area = torch.abs(total) / 2
        return area, selected

    def _oriented_box_intersection_2d(self, corners1: Tensor,
                                    corners2: Tensor) -> Tuple[Tensor, Tensor]:
        intersections, valid_mask = self._box_intersection(corners1, corners2)
        c12, c21 = self._box_in_box(corners1, corners2)
        vertices, mask = self._build_vertices(corners1, corners2, c12, c21,
                                        intersections, valid_mask)
        sorted_indices = self._sort_indices(vertices, mask)
        return self._calculate_area(sorted_indices, vertices)

    def _box2corners(self, box: Tensor) -> Tensor:
        B = box.size()[0]
        x, y, w, h, alpha = box.split([1, 1, 1, 1, 1], dim=-1)
        x4 = box.new_tensor([0.5, -0.5, -0.5, 0.5]).to(box.device)
        x4 = x4 * w
        y4 = box.new_tensor([0.5, 0.5, -0.5, -0.5]).to(box.device)
        y4 = y4 * h
        corners = torch.stack([x4, y4], dim=-1)
        sin = torch.sin(alpha)
        cos = torch.cos(alpha)
        row1 = torch.cat([cos, sin], dim=-1)
        row2 = torch.cat([-sin, cos], dim=-1)
        rot_T = torch.stack([row1, row2], dim=-2)
        rotated = torch.bmm(corners.view([-1, 4, 2]), rot_T.view([-1, 2, 2]))
        rotated = rotated.view([B, -1, 4, 2])
        rotated[..., 0] += x
        rotated[..., 1] += y
        return rotated

    def forward(self, box1: Tensor, box2: Tensor) -> Tensor:
        corners1 = self._box2corners(box1)
        corners2 = self._box2corners(box2)
        intersection, _ = self._oriented_box_intersection_2d(corners1, corners2)
        area1 = box1[:, :, 2] * box1[:, :, 3]
        area2 = box2[:, :, 2] * box2[:, :, 3]
        union = area1 + area2 - intersection
        iou = intersection / union
        return iou

diff_iou_rotated_2d = DiffIouRotated()