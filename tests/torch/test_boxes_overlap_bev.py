import unittest
from math import cos, sin, fabs, atan2
from typing import List
from collections import namedtuple
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import mx_driving.detection
from mx_driving import boxes_overlap_bev

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]

EPS = 1e-8


class Point:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def set(self, _x: float, _y: float):
        self.x = _x
        self.y = _y

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Point(x, y)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Point(x, y)


def cross(p1: Point, p2: Point, p0: Point) -> float:
    if p0 is None:
        return p1.x * p2.y - p1.y * p2.x
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)


def check_rect_cross(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    ret = min(p1.x, p2.x) <= max(q1.x, q2.x) and \
          min(q1.x, q2.x) <= max(p1.x, p2.x) and \
          min(p1.y, p2.y) <= max(q1.y, q2.y) and \
          min(q1.y, q2.y) <= max(p1.y, p2.y)

    return ret


def check_in_box2d(box: List[float], p: Point):
    # params: box (5) [x1, y1, x2, x2, angle]
    MARGIN = 1e-5

    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    # rotate the point in the opposite direction of box
    angle_cos = cos(-box[4])
    angle_sin = sin(-box[4])
    rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * angle_sin + center_x
    rot_y = -(p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos + center_y

    return ((rot_x > box[0] - MARGIN) and (rot_x < box[2] + MARGIN) and
            (rot_y > box[1] - MARGIN) and (rot_y < box[3] + MARGIN))


def intersection(p1: Point, p0: Point, q1: Point, q0: Point):
    ans_point = Point()
    # fast exclusion
    if check_rect_cross(p0, p1, q0, q1) == 0:
        return 0, ans_point

    # check cross standing
    s1 = cross(q0, p1, p0)
    s2 = cross(p1, q1, p0)
    s3 = cross(p0, q1, q0)
    s4 = cross(q1, p1, q0)

    if not (s1 * s2 > 0 and s3 * s4 > 0):
        return 0, ans_point

    # calculate intersection of two lines
    s5 = cross(q1, p1, p0)
    if fabs(s5 - s1) > EPS:
        ans_point.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1)
        ans_point.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1)

    else:
        a0 = p0.y - p1.y
        b0 = p1.x - p0.x
        c0 = p0.x * p1.y - p1.x * p0.y
        a1 = q0.y - q1.y
        b1 = q1.x - q0.x
        c1 = q0.x * q1.y - q1.x * q0.y

        D = a0 * b1 - a1 * b0
        adjusted_D = D if D != 0 else EPS

        ans_point.x = (b0 * c1 - b1 * c0) / adjusted_D
        ans_point.y = (a1 * c0 - a0 * c1) / adjusted_D

    return 1, ans_point


def rotate_around_center(center: Point, angle_cos: float, angle_sin: float, p: Point) -> Point:
    new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * angle_sin + center.x
    new_y = -(p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y
    p.set(new_x, new_y)
    return p


def point_cmp(a: Point, b: Point, center: Point):
    return atan2(a.y - center.y, a.x - center.x) > atan2(b.y - center.y, b.x - center.x)


def box_overlap(box_a: List[float], box_b: List[float]):
    a_x1 = box_a[0]
    a_y1 = box_a[1]
    a_x2 = box_a[2]
    a_y2 = box_a[3]
    a_angle = box_a[4]
    b_x1 = box_b[0]
    b_y1 = box_b[1]
    b_x2 = box_b[2]
    b_y2 = box_b[3]
    b_angle = box_b[4]

    center_a = Point((a_x1 + a_x2) / 2, (a_y1 + a_y2) / 2)
    center_b = Point((b_x1 + b_x2) / 2, (b_y1 + b_y2) / 2)

    box_a_corners = [Point()] * 5
    box_a_corners[0] = Point(a_x1, a_y1)
    box_a_corners[1] = Point(a_x2, a_y1)
    box_a_corners[2] = Point(a_x2, a_y2)
    box_a_corners[3] = Point(a_x1, a_y2)

    box_b_corners = [Point()] * 5
    box_b_corners[0] = Point(b_x1, b_y1)
    box_b_corners[1] = Point(b_x2, b_y1)
    box_b_corners[2] = Point(b_x2, b_y2)
    box_b_corners[3] = Point(b_x1, b_y2)
    # get oriented corners
    a_angle_cos = cos(a_angle)
    a_angle_sin = sin(a_angle)

    b_angle_cos = cos(b_angle)
    b_angle_sin = sin(b_angle)
    for k in range(4):
        rotate_point_a = rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k])
        box_a_corners[k] = rotate_point_a
        rotate_point_b = rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k])
        box_b_corners[k] = rotate_point_b
    box_a_corners[4] = box_a_corners[0]
    box_b_corners[4] = box_b_corners[0]
    cross_points = [Point()] * 16
    poly_center = Point(0, 0)
    cnt = 0
    flag = 0
    for i in range(4):
        for j in range(4):
            flag, ans_point = intersection(box_a_corners[i + 1], box_a_corners[i],
                                           box_b_corners[j + 1], box_b_corners[j])
            cross_points[cnt] = ans_point

            if flag:
                poly_center = poly_center + cross_points[cnt]
                cnt += 1
    # check corners
    for k in range(4):
        if check_in_box2d(box_a, box_b_corners[k]):
            poly_center = poly_center + box_b_corners[k]
            cross_points[cnt] = box_b_corners[k]
            cnt += 1
        if check_in_box2d(box_b, box_a_corners[k]):
            poly_center = poly_center + box_a_corners[k]
            cross_points[cnt] = box_a_corners[k]
            cnt += 1

    if cnt != 0:
        poly_center.x /= cnt
        poly_center.y /= cnt
    # sort the points of polygon

    for j in range(cnt - 1):
        for i in range(cnt - j - 1):
            flag1 = point_cmp(cross_points[i], cross_points[i + 1], poly_center)
            if flag1:
                temp = cross_points[i]
                cross_points[i] = cross_points[i + 1]
                cross_points[i + 1] = temp

    # get the overlap areas
    area = 0
    for k in range(cnt - 1):
        v1 = cross_points[k] - cross_points[0]
        v2 = cross_points[k + 1] - cross_points[0]
        val = cross(v1, v2, None)
        area += val
    return fabs(area) / 2.0


def cpu_boxes_overlap_bev(boxes_a: List[List[float]], boxes_b: List[List[float]]):
    boxes_a_num = boxes_a.shape[0]
    boxes_b_num = boxes_b.shape[0]
    ans = np.zeros((boxes_a_num, boxes_b_num))
    for i in range(boxes_a_num):
        for j in range(boxes_b_num):
            area = box_overlap(boxes_a[i], boxes_b[j])
            ans[i, j] = area
            
    return ans

Inputs = namedtuple('Inputs', ['boxes_a', 'boxes_b'])


class TestBoxesOverlapBev(TestCase):
    np.random.seed(2024)

    def setUp(self):
        self.dtype_list = [torch.float32]
        self.shape_list = [
            [200, 19],
            [200, 60],
            [200, 12],
            [200, 10],
            [10, 200],
            [60, 200],
            [12, 200],
            [10, 200]
        ]
        self.items = [
            [shape, dtype]
            for shape in self.shape_list
            for dtype in self.dtype_list
        ]
        self.test_results = self.gen_results()
    
    def gen_results(self):
        if DEVICE_NAME != 'Ascend910B':
            self.skipTest("OP `BoxesOverlapBev` is only supported on 910B, skipping test data generation!")
        test_results = []
        for shape, dtype in self.items:
            cpu_inputs, npu_inputs = self.gen_inputs(shape, dtype)
            cpu_results = self.cpu_to_exec(cpu_inputs)
            npu_results = self.npu_to_exec(npu_inputs)
            test_results.append((cpu_results, npu_results))
        return test_results
    
    def gen_inputs(self, shape, dtype):
        boxes_a_num, boxes_b_num = shape 
        boxes_a = np.zeros((boxes_a_num, 5))
        boxes_b = np.zeros((boxes_b_num, 5))
        for i in range(boxes_a_num):
            x1 = np.random.uniform(0, 50)
            y1 = np.random.uniform(0, 50)
            x2 = x1 + np.random.uniform(0, 50)
            y2 = y1 + np.random.uniform(0, 50)
            angle = np.random.uniform(0, 1)
            boxes_a[i] = [x1, y1, x2, y2, angle]
        
        for i in range(boxes_b_num):
            x1 = np.random.uniform(0, 50)
            y1 = np.random.uniform(0, 50)
            x2 = x1 + np.random.uniform(0, 50)
            y2 = y1 + np.random.uniform(0, 50)
            angle = np.random.uniform(0, 1)
            boxes_b[i] = [x1, y1, x2, y2, angle]
            
        boxes_a_cpu = boxes_a.astype(np.float32)
        boxes_b_cpu = boxes_b.astype(np.float32)
        
        boxes_a_npu = torch.from_numpy(boxes_a_cpu).npu()
        boxes_b_npu = torch.from_numpy(boxes_b_cpu).npu()
        
        return Inputs(boxes_a_cpu, boxes_b_cpu), \
               Inputs(boxes_a_npu, boxes_b_npu)

    def cpu_to_exec(self, cpu_inputs):
        cpu_boxes_a = cpu_inputs.boxes_a
        cpu_boxes_b = cpu_inputs.boxes_b
        cpu_ans_overlap = cpu_boxes_overlap_bev(cpu_boxes_a, cpu_boxes_b)
        return cpu_ans_overlap.astype(np.float32)

    def npu_to_exec(self, npu_inputs):
        npu_boxes_a = npu_inputs.boxes_a
        npu_boxes_b = npu_inputs.boxes_b
        npu_ans_overlap1 = mx_driving.detection.boxes_overlap_bev(npu_boxes_a, npu_boxes_b)
        npu_ans_overlap1 = npu_ans_overlap1.cpu().float().numpy()
        npu_ans_overlap2 = mx_driving.detection.npu_boxes_overlap_bev(npu_boxes_a, npu_boxes_b)
        npu_ans_overlap2 = npu_ans_overlap2.cpu().float().numpy()
        npu_ans_overlap = boxes_overlap_bev(npu_boxes_a, npu_boxes_b)
        npu_ans_overlap = npu_ans_overlap.cpu().float().numpy()
        return npu_ans_overlap, npu_ans_overlap1, npu_ans_overlap2

    def check_precision(self, actual, expected, rtol=1e-4, atol=1e-4, msg=None):
        if not np.all(np.isclose(actual, expected, rtol=rtol, atol=atol)):
            standardMsg = f'{actual} != {expected} within relative tolerance {rtol}'
            raise AssertionError(msg or standardMsg)

    def test_boxes_overlap_bev(self):
        for cpu_result, npu_results in self.test_results:
            for npu_result in npu_results:
                self.check_precision(cpu_result, npu_result, 1e-4, 1e-4)
 
 
if __name__ == '__main__':
    run_tests()
