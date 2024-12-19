import unittest
from collections import namedtuple

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving.detection
from mx_driving import box_iou_quadri, box_iou_rotated


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class RotatedBox:
    def __init__(self, x_ctr, y_ctr, w, h, a):
        self.x_ctr = x_ctr
        self.y_ctr = y_ctr
        self.w = w
        self.h = h
        self.a = a
        

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, coeff):
        return Point(self.x * coeff, self.y * coeff)


def dot_2d(a: Point, b: Point) -> float:
    return a.x * b.x + a.y * b.y


def cross_2d(a: Point, b: Point) -> float:
    return a.x * b.y - b.x * a.y


def get_rotated_vertices(box):
    theta = box.a
    cosTheta2 = np.cos(theta) * 0.5
    sinTheta2 = np.sin(theta) * 0.5

    pts = [Point(0, 0) for _ in range(4)]

    pts[0].x = box.x_ctr - sinTheta2 * box.h - cosTheta2 * box.w
    pts[0].y = box.y_ctr + cosTheta2 * box.h - sinTheta2 * box.w
    pts[1].x = box.x_ctr + sinTheta2 * box.h - cosTheta2 * box.w
    pts[1].y = box.y_ctr - cosTheta2 * box.h - sinTheta2 * box.w
    pts[2].x = 2 * box.x_ctr - pts[0].x
    pts[2].y = 2 * box.y_ctr - pts[0].y
    pts[3].x = 2 * box.x_ctr - pts[1].x
    pts[3].y = 2 * box.y_ctr - pts[1].y

    return pts


def get_overlap_points(pts1, pts2):
    overlaps = []
    
    vec1 = [pts1[(i + 1) % 4] - pts1[i] for i in range(4)]
    vec2 = [pts2[(i + 1) % 4] - pts2[i] for i in range(4)]
    
    for i in range(4):
        for j in range(4):
            det = cross_2d(vec2[j], vec1[i])
            
            if abs(det) <= 1e-14:
                continue

            vec12 = pts2[j] - pts1[i]
            t1 = cross_2d(vec2[j], vec12) / det
            t2 = cross_2d(vec1[i], vec12) / det

            if 0 <= t1 <= 1 and 0 <= t2 <= 1:
                overlaps.append(pts1[i] + vec1[i] * t1)

    AB = vec2[0]
    DA = vec2[3]
    ABdotAB = dot_2d(AB, AB)
    ADdotAD = dot_2d(DA, DA)

    for i in range(4):
        AP = pts1[i] - pts2[0]
        APdotAB = dot_2d(AP, AB)
        APdotAD = -dot_2d(AP, DA)

        if 0 <= APdotAB <= ABdotAB and 0 <= APdotAD <= ADdotAD:
            overlaps.append(pts1[i])

    AB = vec1[0]
    DA = vec1[3]
    ABdotAB = dot_2d(AB, AB)
    ADdotAD = dot_2d(DA, DA)

    for i in range(4):
        AP = pts2[i] - pts1[0]
        APdotAB = dot_2d(AP, AB)
        APdotAD = -dot_2d(AP, DA)

        if 0 <= APdotAB <= ABdotAB and 0 <= APdotAD <= ADdotAD:
            overlaps.append(pts2[i])

    return len(overlaps), overlaps


def convex_hull_graham(points, num_in, shift_to_zero=False):
    t = 0
    for i in range(1, num_in):
        if points[i].y < points[t].y or (points[i].y == points[t].y and points[i].x < points[t].x):
            t = i
    start = points[t]

    q = [p - start for p in points]

    q[0], q[t] = q[t], q[0]
    
    dist = [dot_2d(p, p) for p in q]
    
    for i in range(1, num_in - 1):
        for j in range(i + 1, num_in):
            cross_product = cross_2d(q[i], q[j])
            if (cross_product < -1e-6) or (abs(cross_product) < 1e-6 and dist[i] > dist[j]):
                q[i], q[j] = q[j], q[i]
                dist[i], dist[j] = dist[j], dist[i]

    k = 1
    while k < num_in and dist[k] <= 1e-8:
        k += 1
    if k == num_in:
        return 1, [points[t]]

    q[1] = q[k]
    
    m = 2
    for i in range(k + 1, num_in):
        while m > 1 and cross_2d(q[i] - q[m - 2], q[m - 1] - q[m - 2]) >= 0:
            m -= 1
        q[m] = q[i]
        m += 1

    if not shift_to_zero:
        for i in range(m):
            q[i] = q[i] + start
    
    return m, q[:m]


def quadri_box_area(q):
    area = 0.0
    
    for i in range(1, 3):
        area += abs(cross_2d(q[i] - q[0], q[i + 1] - q[0]))

    return area / 2.0


def polygon_area(points, m):
    if m <= 2:
        return 0

    area = 0.0
    
    for i in range(1, m - 1):
        area += abs(cross_2d(points[i] - points[0], points[i + 1] - points[0]))

    return area / 2.0


def rotated_boxes_overlap(box1, box2):
    pts1 = get_rotated_vertices(box1)
    pts2 = get_rotated_vertices(box2)

    num, overlap_pts = get_overlap_points(pts1, pts2)
    
    if len(overlap_pts) <= 2:
        return 0.0

    num_convex, ordered_pts = convex_hull_graham(overlap_pts, num, True)

    return polygon_area(ordered_pts, num_convex)


def quadri_boxes_overlap(pts1, pts2):
    num, overlap_pts = get_overlap_points(pts1, pts2)
    
    if num <= 2:
        return 0.0

    num_convex, ordered_pts = convex_hull_graham(overlap_pts, num, True)

    return polygon_area(ordered_pts, num_convex)


def single_box_iou_rotated(box1_raw, box2_raw, mode_flag):
    center_shift_x = (box1_raw[0] + box2_raw[0]) / 2.0
    center_shift_y = (box1_raw[1] + box2_raw[1]) / 2.0
    
    box1 = RotatedBox(box1_raw[0] - center_shift_x, box1_raw[1] - center_shift_y, 
                      box1_raw[2], box1_raw[3], box1_raw[4])
    box2 = RotatedBox(box2_raw[0] - center_shift_x, box2_raw[1] - center_shift_y, 
                      box2_raw[2], box2_raw[3], box2_raw[4])
    
    area1 = box1.w * box1.h
    area2 = box2.w * box2.h
    
    if area1 < 1e-14 or area2 < 1e-14:
        return 0.0

    overlap = rotated_boxes_overlap(box1, box2)
    
    if mode_flag == 0:
        baseS = area1 + area2 - overlap
    elif mode_flag == 1:
        baseS = area1
    
    iou = overlap / baseS
    return iou


def single_box_iou_quadri(pts1_raw, pts2_raw, mode_flag):
    center_shift_x = (pts1_raw[0] + pts2_raw[0] + pts1_raw[2] + pts2_raw[2] +
                      pts1_raw[4] + pts2_raw[4] + pts1_raw[6] + pts2_raw[6]) / 8.0
    center_shift_y = (pts1_raw[1] + pts2_raw[1] + pts1_raw[3] + pts2_raw[3] +
                      pts1_raw[5] + pts2_raw[5] + pts1_raw[7] + pts2_raw[7]) / 8.0

    pts1 = [Point(pts1_raw[i] - center_shift_x, pts1_raw[i + 1] - center_shift_y) for i in range(0, 8, 2)]
    pts2 = [Point(pts2_raw[i] - center_shift_x, pts2_raw[i + 1] - center_shift_y) for i in range(0, 8, 2)]
    
    area1 = quadri_box_area(pts1)
    area2 = quadri_box_area(pts2)
    
    if area1 < 1e-14 or area2 < 1e-14:
        return 0.0
    
    overlap = quadri_boxes_overlap(pts1, pts2)
    
    if mode_flag == 0:
        baseS = area1 + area2 - overlap
    elif mode_flag == 1:
        baseS = area1
    
    iou = overlap / baseS
    return iou


def cpu_box_iou_rotated_unaligned(boxes1, boxes2, mode="iou"):
    assert mode in ["iou", "iof"], "mode must be 'iou' or 'iof'"
    
    num1 = boxes1.shape[0]
    num2 = boxes2.shape[0]
    
    ious = np.zeros((num1, num2), dtype=np.float32)
    
    for i in range(num1):
        for j in range(num2):
            ious[i, j] = single_box_iou_rotated(boxes1[i], boxes2[j], mode_flag=0 if mode == "iou" else 1)
    
    return ious


def cpu_box_iou_rotated_aligned(boxes1, boxes2, mode="iou"):
    assert mode in ["iou", "iof"], "mode must be 'iou' or 'iof'"
    
    num1 = boxes1.shape[0]
    num2 = boxes2.shape[0]
    num = min(num1, num2)
    
    ious = np.zeros((num,), dtype=np.float32)
    
    for i in range(num):
        ious[i] = single_box_iou_rotated(boxes1[i], boxes2[i], mode_flag=0 if mode == "iou" else 1)
    
    return ious


@golden_data_cache(__file__)
def cpu_box_iou_rotated(boxes1, boxes2, mode="iou", aligned=False):
    if aligned:
        return cpu_box_iou_rotated_aligned(boxes1, boxes2, mode)
    else:
        return cpu_box_iou_rotated_unaligned(boxes1, boxes2, mode)


def cpu_box_iou_quadri_unaligned(boxes1, boxes2, mode="iou"):
    assert mode in ["iou", "iof"], "mode must be 'iou' or 'iof'"
    
    num1 = boxes1.shape[0]
    num2 = boxes2.shape[0]
    
    ious = np.zeros((num1, num2), dtype=np.float32)
    
    for i in range(num1):
        for j in range(num2):
            ious[i, j] = single_box_iou_quadri(boxes1[i], boxes2[j], mode_flag=0 if mode == "iou" else 1)
    
    return ious


def cpu_box_iou_quadri_aligned(boxes1, boxes2, mode="iou"):
    assert mode in ["iou", "iof"], "mode must be 'iou' or 'iof'"
    
    num1 = boxes1.shape[0]
    num2 = boxes2.shape[0]
    num = min(num1, num2)
    
    ious = np.zeros((num,), dtype=np.float32)
    
    for i in range(num):
        ious[i] = single_box_iou_quadri(boxes1[i], boxes2[i], mode_flag=0 if mode == "iou" else 1)
    
    return ious


@golden_data_cache(__file__)
def cpu_box_iou_quadri(boxes1, boxes2, mode="iou", aligned=False):
    if aligned:
        return cpu_box_iou_quadri_aligned(boxes1, boxes2, mode)
    else:
        return cpu_box_iou_quadri_unaligned(boxes1, boxes2, mode)
    
    
@golden_data_cache(__file__)
def gen_boxes_rotated(boxes_num):
    boxes = []
    for _ in range(boxes_num):
        x_center = np.random.uniform(-100, 100)
        y_center = np.random.uniform(-100, 100)
        width = np.random.uniform(10, 50)
        height = np.random.uniform(10, 50)
        angle = np.random.uniform(-np.pi, np.pi)
        boxes.append([x_center, y_center, width, height, angle])
    return np.array(boxes)


def boxes_to_pts(boxes):
    pts = []
    for box in boxes:
        x_center, y_center, width, height, angle = box
        cosTheta2 = np.cos(angle) * 0.5
        sinTheta2 = np.sin(angle) * 0.5
        x1 = x_center - sinTheta2 * height - cosTheta2 * width
        y1 = y_center + cosTheta2 * height - sinTheta2 * width
        x2 = x_center + sinTheta2 * height - cosTheta2 * width
        y2 = y_center - cosTheta2 * height - sinTheta2 * width
        x3 = 2 * x_center - x1
        y3 = 2 * y_center - y1
        x4 = 2 * x_center - x2
        y4 = 2 * y_center - y2
        x1 = x1 - np.random.uniform(0, 5)
        y1 = y1 + np.random.uniform(0, 5)
        x2 = x2 - np.random.uniform(0, 5)
        y2 = y2 - np.random.uniform(0, 5)
        x3 = x3 + np.random.uniform(0, 5)
        y3 = y3 - np.random.uniform(0, 5)
        x4 = x4 + np.random.uniform(0, 5)
        y4 = y4 + np.random.uniform(0, 5)
        pts.append([x1, y1, x2, y2, x3, y3, x4, y4])
    return np.array(pts)


@golden_data_cache(__file__)
def gen_boxes_quadri(boxes_num):
    boxes = gen_boxes_rotated(boxes_num)
    boxes = boxes_to_pts(boxes)
    return boxes


Inputs = namedtuple('Inputs', ['boxes_a', 'boxes_b', 'mode', 'aligned'])


class TestBoxIouRotated(TestCase):
    np.random.seed(2024)

    def setUp(self):
        dtype_list = [torch.float32]
        mode_list = ['iou', 'iof']
        
        # unaligned
        shape_list = [
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
            [shape, dtype, mode, False]
            for shape in shape_list
            for dtype in dtype_list
            for mode in mode_list
        ]
        
        # aligned
        shape_list = [
            [20, 20],
            [200, 200],
            [2000, 2000]
        ]
        self.items += [
            [shape, dtype, mode, True]
            for shape in shape_list
            for dtype in dtype_list
            for mode in mode_list
        ]
        
        self.test_results = self.gen_results()
    
    def gen_results(self):
        if DEVICE_NAME != 'Ascend910B':
            self.skipTest("OP `BoxIou` is only supported on 910B, skipping test data generation!")
        test_results = []
        for shape, dtype, mode, aligned in self.items:
            cpu_inputs, npu_inputs = self.gen_inputs(shape, dtype, mode, aligned)
            cpu_results = self.cpu_to_exec(cpu_inputs)
            npu_results = self.npu_to_exec(npu_inputs)
            test_results.append((cpu_results, npu_results))
        return test_results
    
    def gen_inputs(self, shape, dtype, mode, aligned):
        boxes_a_num, boxes_b_num = shape 
        boxes_a = gen_boxes_rotated(boxes_a_num)
        boxes_b = gen_boxes_rotated(boxes_b_num)
            
        boxes_a_cpu = boxes_a.astype(np.float32)
        boxes_b_cpu = boxes_b.astype(np.float32)
        
        boxes_a_npu = torch.from_numpy(boxes_a_cpu).npu()
        boxes_b_npu = torch.from_numpy(boxes_b_cpu).npu()
        
        return Inputs(boxes_a_cpu, boxes_b_cpu, mode, aligned), \
               Inputs(boxes_a_npu, boxes_b_npu, mode, aligned)

    def cpu_to_exec(self, cpu_inputs):
        cpu_boxes_a = cpu_inputs.boxes_a
        cpu_boxes_b = cpu_inputs.boxes_b
        mode = cpu_inputs.mode
        aligned = cpu_inputs.aligned
        cpu_ans_ious = cpu_box_iou_rotated(cpu_boxes_a, cpu_boxes_b, mode, aligned)
        return cpu_ans_ious.astype(np.float32)

    def npu_to_exec(self, npu_inputs):
        npu_boxes_a = npu_inputs.boxes_a
        npu_boxes_b = npu_inputs.boxes_b
        mode = npu_inputs.mode
        aligned = npu_inputs.aligned
        npu_ans_ious = box_iou_rotated(npu_boxes_a, npu_boxes_b, mode, aligned)
        return npu_ans_ious.cpu().float().numpy()

    def check_precision(self, actual, expected, rtol=1e-4, atol=1e-4, msg=None):
        if not np.all(np.isclose(actual, expected, rtol=rtol, atol=atol)):
            standardMsg = f'{actual} != {expected} within relative tolerance {rtol}'
            raise AssertionError(msg or standardMsg)

    def test_box_iou_rotated(self):
        for cpu_results, npu_results in self.test_results:
            self.check_precision(cpu_results, npu_results, 1e-4, 1e-4)


class TestBoxIouQuadri(TestCase):
    np.random.seed(2024)

    def setUp(self):
        dtype_list = [torch.float32]
        mode_list = ['iou', 'iof']
        
        # unaligned
        shape_list = [
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
            [shape, dtype, mode, False]
            for shape in shape_list
            for dtype in dtype_list
            for mode in mode_list
        ]
        
        # aligned
        shape_list = [
            [20, 20],
            [200, 200],
            [2000, 2000]
        ]
        self.items += [
            [shape, dtype, mode, True]
            for shape in shape_list
            for dtype in dtype_list
            for mode in mode_list
        ]
        
        self.test_results = self.gen_results()
    
    def gen_results(self):
        if DEVICE_NAME != 'Ascend910B':
            self.skipTest("OP `BoxIou` is only supported on 910B, skipping test data generation!")
        test_results = []
        for shape, dtype, mode, aligned in self.items:
            cpu_inputs, npu_inputs = self.gen_inputs(shape, dtype, mode, aligned)
            cpu_results = self.cpu_to_exec(cpu_inputs)
            npu_results = self.npu_to_exec(npu_inputs)
            test_results.append((cpu_results, npu_results))
        return test_results
    
    def gen_inputs(self, shape, dtype, mode, aligned):
        boxes_a_num, boxes_b_num = shape 
        boxes_a = gen_boxes_quadri(boxes_a_num)
        boxes_b = gen_boxes_quadri(boxes_b_num)
        
        boxes_a_cpu = boxes_a.astype(np.float32)
        boxes_b_cpu = boxes_b.astype(np.float32)
        
        boxes_a_npu = torch.from_numpy(boxes_a_cpu).npu()
        boxes_b_npu = torch.from_numpy(boxes_b_cpu).npu()
        
        return Inputs(boxes_a_cpu, boxes_b_cpu, mode, aligned), \
               Inputs(boxes_a_npu, boxes_b_npu, mode, aligned)

    def cpu_to_exec(self, cpu_inputs):
        cpu_boxes_a = cpu_inputs.boxes_a
        cpu_boxes_b = cpu_inputs.boxes_b
        mode = cpu_inputs.mode
        aligned = cpu_inputs.aligned
        cpu_ans_ious = cpu_box_iou_quadri(cpu_boxes_a, cpu_boxes_b, mode, aligned)
        return cpu_ans_ious.astype(np.float32)

    def npu_to_exec(self, npu_inputs):
        npu_boxes_a = npu_inputs.boxes_a
        npu_boxes_b = npu_inputs.boxes_b
        mode = npu_inputs.mode
        aligned = npu_inputs.aligned
        npu_ans_ious1 = mx_driving.detection.box_iou_quadri(npu_boxes_a, npu_boxes_b, mode, aligned)
        npu_ans_ious1 = npu_ans_ious1.cpu().float().numpy()
        npu_ans_ious = box_iou_quadri(npu_boxes_a, npu_boxes_b, mode, aligned)
        npu_ans_ious = npu_ans_ious.cpu().float().numpy()
        return npu_ans_ious, npu_ans_ious1

    def check_precision(self, actual, expected, rtol=1e-4, atol=1e-4, msg=None):
        if not np.all(np.isclose(actual, expected, rtol=rtol, atol=atol)):
            standardMsg = f'{actual} != {expected} within relative tolerance {rtol}'
            raise AssertionError(msg or standardMsg)

    def test_box_iou_quadri(self):
        for cpu_result, npu_results in self.test_results:
            for npu_result in npu_results:
                self.check_precision(cpu_result, npu_result, 1e-4, 1e-4)


if __name__ == '__main__':
    run_tests()
