from typing import Tuple
import torch
import numpy as np
from torch import Tensor
from torch.autograd import Function
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import mx_driving

torch.npu.set_device(2)

EPSILON = 1e-6
INF = torch.inf


class DiffIouRotatedSortVerticesGloden(Function):

    @staticmethod
    def forward(ctx, vertices, mask, num_valid):
        B, N, _, _ = vertices.shape
        vertices[~mask] = INF
        min_point_idx = vertices[..., 1].argmin(-1)
        B_idx = torch.arange(B).reshape(-1, 1)
        N_idx = torch.arange(N).reshape(1, -1)
        min_point_pos = vertices[B_idx, N_idx, min_point_idx]
        vertices -= min_point_pos.unsqueeze(-2)
        vertices_radian = torch.atan2(vertices[..., 1], (vertices[..., 0]))
        vertices_radian[B_idx, N_idx, min_point_idx] = -INF
        vertices_radian[~mask] = INF
        sorted_idx = torch.argsort(vertices_radian, dim=-1, descending=False)[:, :, :9]
        select_idx = torch.arange(9, device=vertices.device).repeat(1, N * B).reshape(B, N, -1) < num_valid.unsqueeze(-1)
        sorted_idx = torch.where(select_idx, sorted_idx, sorted_idx[B_idx, N_idx, 0:1])
        return sorted_idx

    @staticmethod
    def backward(ctx, gradout):
        return ()


def box_intersection(corners1: Tensor,
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
    u = -denumerator_u / (numerator + EPSILON)
    mask_u = (u >= 0) & (u <= 1)
    mask = mask_t * mask_u
    t = denumerator_t / (numerator)
    intersections = torch.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)],
                                dim=-1)
    intersections = intersections * mask.float().unsqueeze(-1)
    return intersections, mask


def box1_in_box2(corners1: Tensor, corners2: Tensor) -> Tensor:
    corners_related = corners2[:, :, None, :, :] - corners1[:, :, :, None, :]
    x = corners_related[..., 0]
    y = corners_related[..., 1]
    line_to_x = x[..., [1, 2, 3, 0]]
    line_to_y = y[..., [1, 2, 3, 0]]
    mask1 = (line_to_y > 0) ^ (y > 0)
    mask2 = x - y * (line_to_x - x) / (line_to_y - y) > 0
    return (mask1 & mask2).sum(-1) % 2 == 1


def box_in_box(corners1: Tensor, corners2: Tensor) -> Tuple[Tensor, Tensor]:
    c1_in_2 = box1_in_box2(corners1, corners2)
    c2_in_1 = box1_in_box2(corners2, corners1)
    return c1_in_2, c2_in_1


def build_vertices(corners1: Tensor, corners2: Tensor, c1_in_2: Tensor,
                   c2_in_1: Tensor, intersections: Tensor,
                   valid_mask: Tensor) -> Tuple[Tensor, Tensor]:
    B = corners1.size()[0]
    N = corners1.size()[1]
    vertices = torch.cat(
        [corners1, corners2,
         intersections.view([B, N, -1, 2])], dim=2)
    mask = torch.cat([c1_in_2, c2_in_1, valid_mask.view([B, N, -1])], dim=2)
    return vertices, mask


def sort_indices(vertices: Tensor, mask: Tensor) -> Tensor:
    num_valid = torch.sum(mask.int(), dim=2).int()
    mean = torch.sum(
        vertices * mask.float().unsqueeze(-1), dim=2,
        keepdim=True) / num_valid.unsqueeze(-1).unsqueeze(-1)
    vertices_normalized = vertices - mean
    return DiffIouRotatedSortVerticesGloden.apply(vertices_normalized, mask, num_valid).long()


def calculate_area(idx_sorted: Tensor,
                   vertices: Tensor) -> Tuple[Tensor, Tensor]:
    idx_ext = idx_sorted.unsqueeze(-1).repeat([1, 1, 1, 2])
    selected = torch.gather(vertices, 2, idx_ext)
    total = selected[:, :, 0:-1, 0] * selected[:, :, 1:, 1] \
        - selected[:, :, 0:-1, 1] * selected[:, :, 1:, 0]
    total = torch.sum(total, dim=2)
    area = torch.abs(total) / 2
    return area, selected


def oriented_box_intersection_2d(corners1: Tensor,
                                 corners2: Tensor) -> Tuple[Tensor, Tensor]:
    intersections, valid_mask = box_intersection(corners1, corners2)
    c12, c21 = box_in_box(corners1, corners2)
    vertices, mask = build_vertices(corners1, corners2, c12, c21,
                                    intersections, valid_mask)
    sorted_indices = sort_indices(vertices, mask)
    return calculate_area(sorted_indices, vertices)


def box2corners(box: Tensor) -> Tensor:
    B = box.size()[0]
    x, y, w, h, alpha = box.split([1, 1, 1, 1, 1], dim=-1)
    x4 = box.new_tensor([0.5, -0.5, -0.5, 0.5]).float().to(box.device)
    x4 = x4 * w
    y4 = box.new_tensor([0.5, 0.5, -0.5, -0.5]).float().to(box.device)
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


def diff_iou_rotated_2d_gloden(box1: Tensor, box2: Tensor) -> Tensor:
    corners1 = box2corners(box1)
    corners2 = box2corners(box2)
    intersection, _ = oriented_box_intersection_2d(corners1,
                                                   corners2)  # (B, N)
    area1 = box1[:, :, 2] * box1[:, :, 3]
    area2 = box2[:, :, 2] * box2[:, :, 3]
    union = area1 + area2 - intersection
    iou = intersection / union
    return iou


class TestDiffIouRoatated(TestCase):

    def gen_boxes_rotated(self, B, N,
        center_uniform_left, center_uniform_right,
        width_uniform_left, width_uniform_right,
        height_uniform_left, height_uniform_right):
        
        boxes = []
        for _ in range(B * N):
            x_center = np.random.uniform(center_uniform_left, center_uniform_right)
            y_center = np.random.uniform(center_uniform_left, center_uniform_right)
            width = np.random.uniform(width_uniform_left, width_uniform_right)
            height = np.random.uniform(height_uniform_left, height_uniform_right)
            angle = np.random.uniform(-np.pi, np.pi)
            boxes.append([x_center, y_center, width, height, angle])

        return torch.from_numpy(np.array(boxes).reshape(B, N, 5).astype("float32"))

    def test_with_config(self, B, N,
        center_uniform_left, center_uniform_right,
        width_uniform_left, width_uniform_right,
        height_uniform_left, height_uniform_right):

        box1 = self.gen_boxes_rotated(B, N,
            center_uniform_left, center_uniform_right,
            width_uniform_left, width_uniform_right,
            height_uniform_left, height_uniform_right).npu()
        
        box2 = self.gen_boxes_rotated(B, N,
            center_uniform_left, center_uniform_right,
            width_uniform_left, width_uniform_right,
            height_uniform_left, height_uniform_right).npu()

        res = mx_driving.diff_iou_rotated_2d(box1, box2).cpu()
        res_cpu = diff_iou_rotated_2d_gloden(box1, box2)

        self.assertRtolEqual(res, res_cpu)

    def normal_test_case(self):
        self.test_with_config(5, 37, -10, 10, 10, 10, 10, 10)
        self.test_with_config(7, 55, -10, 10, 10, 100, 10, 100)
        self.test_with_config(15, 101, -10, 10, 10, 100, 10, 100)
        self.test_with_config(8, 256, -10, 10, 10, 100, 10, 100)
        
    def max_border_shape_test_case(self):
        self.test_with_config(1024, 2048, -10, 10, 10, 100, 10, 100)

    def min_border_shape_test_case(self):
        self.test_with_config(1, 1, -10, 10, 10, 100, 10, 100)

    def min_box_test_case(self):
        self.test_with_config(32, 32, -10, 10, 1e-5, 1e-5, 1e-5, 1e-5)

    def max_box_test_case(self):
        self.test_with_config(32, 32, -200, 200, 1000, 1000, 1000, 1000)

    def test_diff_rotated_iou_2d(self):
        self.normal_test_case()
        self.min_border_shape_test_case()
        self.min_box_test_case()
        self.max_box_test_case()
        self.max_border_shape_test_case()
        

if __name__ == "__main__":
    seed = 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    TestDiffIouRoatated().test_diff_rotated_iou_2d()
    