from collections import namedtuple
from math import atan2, cos, fabs, sin
from typing import List

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests
import mx_driving
import mx_driving._C
import mx_driving.detection


EPS = 1e-8


class Box:
    def __init__(self):
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.x_center = 0
        self.y_center = 0
        self.dx = 0
        self.dy = 0
        self.angle = 0


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


def point_cmp(a: Point, b: Point, center: Point):
    return atan2(a.y - center.y, a.x - center.x) > atan2(b.y - center.y, b.x - center.x)


def check_in_box2d(box, p, clockwise, margin):
    # rotate the point in the opposite direction of box
    center_point = Point(box.x_center, box.y_center)
    angle_cos = cos(-box.angle)
    angle_sin = sin(-box.angle)
    
    rot_point = Point(p.x, p.y)
    rotate_around_center(center_point, angle_cos, angle_sin, rot_point, clockwise)

    return ((rot_point.x > box.x1 - margin) and (rot_point.x < box.x2 + margin) and
            (rot_point.y > box.y1 - margin) and (rot_point.y < box.y2 + margin))


def rotate_around_center(center, angle_cos, angle_sin, p, clockwise):
    if clockwise:
        new_x = (p.x - center.x) * angle_cos - (p.y - center.y) * angle_sin + center.x
        new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y
    else:
        new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * angle_sin + center.x
        new_y = -(p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y
    p.set(new_x, new_y)


def box_overlap(box_a, box_b, clockwise, margin):
    a_angle = box_a.angle
    b_angle = box_b.angle

    center_a = Point(box_a.x_center, box_a.y_center)
    center_b = Point(box_b.x_center, box_b.y_center)

    box_a_corners = [Point()] * 5
    box_a_corners[0] = Point(box_a.x1, box_a.y1)
    box_a_corners[1] = Point(box_a.x2, box_a.y1)
    box_a_corners[2] = Point(box_a.x2, box_a.y2)
    box_a_corners[3] = Point(box_a.x1, box_a.y2)

    box_b_corners = [Point()] * 5
    box_b_corners[0] = Point(box_b.x1, box_b.y1)
    box_b_corners[1] = Point(box_b.x2, box_b.y1)
    box_b_corners[2] = Point(box_b.x2, box_b.y2)
    box_b_corners[3] = Point(box_b.x1, box_b.y2)

    # get oriented corners
    a_angle_cos = cos(a_angle)
    a_angle_sin = sin(a_angle)
    b_angle_cos = cos(b_angle)
    b_angle_sin = sin(b_angle)

    for k in range(4):
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k], clockwise)
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k], clockwise)
    box_a_corners[4] = box_a_corners[0]
    box_b_corners[4] = box_b_corners[0]

    # get intersection points of line segments
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
        if check_in_box2d(box_a, box_b_corners[k], clockwise, margin):
            poly_center = poly_center + box_b_corners[k]
            cross_points[cnt] = box_b_corners[k]
            cnt += 1
        if check_in_box2d(box_b, box_a_corners[k], clockwise, margin):
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


def parse_box(box, inp_format, r_unit):
    new_box = Box()

    if inp_format == "xyxyr":
        new_box.x1 = box[0]
        new_box.y1 = box[1]
        new_box.x2 = box[2]
        new_box.y2 = box[3]
        new_box.angle = box[4]
        new_box.x_center = (new_box.x1 + new_box.x2) / 2
        new_box.y_center = (new_box.y1 + new_box.y2) / 2
        new_box.dx = abs(new_box.x2 - new_box.x1)
        new_box.dy = abs(new_box.y2 - new_box.y1)
    elif inp_format == "xywhr":
        new_box.x_center = box[0]
        new_box.y_center = box[1]
        new_box.dx = box[2]
        new_box.dy = box[3]
        new_box.angle = box[4]
        new_box.x1 = new_box.x_center - new_box.dx / 2
        new_box.y1 = new_box.y_center - new_box.dy / 2
        new_box.x2 = new_box.x_center + new_box.dx / 2
        new_box.y2 = new_box.y_center + new_box.dy / 2
    elif inp_format == "xyzxyzr":
        new_box.x1 = box[0]
        new_box.y1 = box[1]
        new_box.x2 = box[3]
        new_box.y2 = box[4]
        new_box.angle = box[6]
        new_box.x_center = (new_box.x1 + new_box.x2) / 2
        new_box.y_center = (new_box.y1 + new_box.y2) / 2
        new_box.dx = abs(new_box.x2 - new_box.x1)
        new_box.dy = abs(new_box.y2 - new_box.y1)
    elif inp_format == "xyzwhdr":
        new_box.x_center = box[0]
        new_box.y_center = box[1]
        new_box.dx = box[3]
        new_box.dy = box[4]
        new_box.angle = box[6]
        new_box.x1 = new_box.x_center - new_box.dx / 2
        new_box.y1 = new_box.y_center - new_box.dy / 2
        new_box.x2 = new_box.x_center + new_box.dx / 2
        new_box.y2 = new_box.y_center + new_box.dy / 2

    if r_unit == "degree":
        new_box.angle = new_box.angle * np.pi / 180

    return new_box


def single_compute(box_a, box_b, attrs):
    inp_format, r_unit, clockwise, mode, margin = attrs
    box_a = parse_box(box_a, inp_format, r_unit)
    box_b = parse_box(box_b, inp_format, r_unit)
    
    overlap = box_overlap(box_a, box_b, clockwise, margin)
    if mode == "iou":
        area_a = box_a.dx * box_a.dy
        area_b = box_b.dx * box_b.dy
        iou = overlap / max(area_a + area_b - overlap, EPS)
        return iou
    if mode == "iof":
        area_a = box_a.dx * box_a.dy
        iof = overlap / max(area_a, EPS)
        return iof
    return overlap


@golden_data_cache(__file__)
def cpu_boxes_overlap_bev(boxes_a, boxes_b, attrs):
    aligned = attrs[0]
    boxes_a_num = boxes_a.shape[0]
    boxes_b_num = boxes_b.shape[0]
    boxes_min_num = min(boxes_a_num, boxes_b_num)
    
    if aligned:
        ans = np.zeros((boxes_a_num))
        for i in range(boxes_min_num):
            ans[i] = single_compute(boxes_a[i], boxes_b[i], attrs[1:])
    else:
        ans = np.zeros((boxes_a_num, boxes_b_num))
        for i in range(boxes_a_num):
            for j in range(boxes_b_num):
                ans[i, j] = single_compute(boxes_a[i], boxes_b[j], attrs[1:])
    return ans


@golden_data_cache(__file__)
def cpu_gen_boxes(shape, inp_format, r_unit):
    boxes_a_num, boxes_b_num = shape

    if inp_format == "xyxyr":
        boxes_a = np.zeros((boxes_a_num, 5))
        boxes_b = np.zeros((boxes_b_num, 5))
        for i in range(boxes_a_num):
            x1 = np.random.uniform(-5, 5)
            y1 = np.random.uniform(-5, 5)
            x2 = np.random.uniform(-5, 5)
            y2 = np.random.uniform(-5, 5)
            angle = np.random.uniform(-np.pi, np.pi)
            angle = angle if r_unit == "radian" else angle * 180 / np.pi
            boxes_a[i] = [x1, y1, x2, y2, angle]
        for i in range(boxes_b_num):
            x1 = np.random.uniform(-5, 5)
            y1 = np.random.uniform(-5, 5)
            x2 = np.random.uniform(-5, 5)
            y2 = np.random.uniform(-5, 5)
            angle = np.random.uniform(-np.pi, np.pi)
            angle = angle if r_unit == "radian" else angle * 180 / np.pi
            boxes_b[i] = [x1, y1, x2, y2, angle]
    elif inp_format == "xywhr":
        boxes_a = np.zeros((boxes_a_num, 5))
        boxes_b = np.zeros((boxes_b_num, 5))
        for i in range(boxes_a_num):
            x_center = np.random.uniform(-5, 5)
            y_center = np.random.uniform(-5, 5)
            dx = np.random.uniform(0, 5)
            dy = np.random.uniform(0, 5)
            angle = np.random.uniform(-np.pi, np.pi)
            angle = angle if r_unit == "radian" else angle * 180 / np.pi
            boxes_a[i] = [x_center, y_center, dx, dy, angle]
        for i in range(boxes_b_num):
            x_center = np.random.uniform(-5, 5)
            y_center = np.random.uniform(-5, 5)
            dx = np.random.uniform(0, 5)
            dy = np.random.uniform(0, 5)
            angle = np.random.uniform(-np.pi, np.pi)
            angle = angle if r_unit == "radian" else angle * 180 / np.pi
            boxes_b[i] = [x_center, y_center, dx, dy, angle]
    elif inp_format == "xyzxyzr":
        boxes_a = np.zeros((boxes_a_num, 7))
        boxes_b = np.zeros((boxes_b_num, 7))
        for i in range(boxes_a_num):
            x1 = np.random.uniform(-5, 5)
            y1 = np.random.uniform(-5, 5)
            z1 = np.random.uniform(-5, 5)
            x2 = np.random.uniform(-5, 5)
            y2 = np.random.uniform(-5, 5)
            z2 = np.random.uniform(-5, 5)
            angle = np.random.uniform(-np.pi, np.pi)
            angle = angle if r_unit == "radian" else angle * 180 / np.pi
            boxes_a[i] = [x1, y1, z1, x2, y2, z2, angle]
        for i in range(boxes_b_num):
            x1 = np.random.uniform(-5, 5)
            y1 = np.random.uniform(-5, 5)
            z1 = np.random.uniform(-5, 5)
            x2 = np.random.uniform(-5, 5)
            y2 = np.random.uniform(-5, 5)
            z2 = np.random.uniform(-5, 5)
            angle = np.random.uniform(-np.pi, np.pi)
            angle = angle if r_unit == "radian" else angle * 180 / np.pi
            boxes_b[i] = [x1, y1, z1, x2, y2, z2, angle]
    elif inp_format == "xyzwhdr":
        boxes_a = np.zeros((boxes_a_num, 7))
        boxes_b = np.zeros((boxes_b_num, 7))
        for i in range(boxes_a_num):
            x_center = np.random.uniform(-5, 5)
            y_center = np.random.uniform(-5, 5)
            z_center = np.random.uniform(-5, 5)
            dx = np.random.uniform(0, 5)
            dy = np.random.uniform(0, 5)
            dz = np.random.uniform(0, 5)
            angle = np.random.uniform(-np.pi, np.pi)
            angle = angle if r_unit == "radian" else angle * 180 / np.pi
            boxes_a[i] = [x_center, y_center, z_center, dx, dy, dz, angle]
        for i in range(boxes_b_num):
            x_center = np.random.uniform(-5, 5)
            y_center = np.random.uniform(-5, 5)
            z_center = np.random.uniform(-5, 5)
            dx = np.random.uniform(0, 5)
            dy = np.random.uniform(0, 5)
            dz = np.random.uniform(0, 5)
            angle = np.random.uniform(-np.pi, np.pi)
            angle = angle if r_unit == "radian" else angle * 180 / np.pi
            boxes_b[i] = [x_center, y_center, z_center, dx, dy, dz, angle]

    boxes_a_cpu = boxes_a.astype(np.float32)
    boxes_b_cpu = boxes_b.astype(np.float32)

    return boxes_a_cpu, boxes_b_cpu


Inputs = namedtuple('Inputs', ['boxes_a', 'boxes_b'])


class TestBoxesOverlapBev(TestCase):
    np.random.seed(2024)

    def setUp(self):
        self.format_dict = {
            "xyxyr": 0,
            "xywhr": 1,
            "xyzxyzr": 2,
            "xyzwhdr": 3
        }
        self.unit_dict = {
            "radian": 0,
            "degree": 1
        }
        self.mode_dict = {
            "overlap": 0,
            "iou": 1,
            "iof": 2
        }
    
    def gen_results(self, cases, ut_name):
        test_results = []
        for shape, inp_format, r_unit, clockwise, mode, aligned, margin in cases:
            cpu_inputs, npu_inputs = self.gen_inputs(shape, inp_format, r_unit)
            attrs = [aligned, inp_format, r_unit, clockwise, mode, margin]
            cpu_results = self.cpu_to_exec(cpu_inputs, attrs)
            npu_results = self.npu_to_exec(npu_inputs, attrs, ut_name)
            test_results.append((cpu_results, npu_results))
        return test_results
    
    def gen_inputs(self, shape, inp_format, r_unit):
        boxes_a_cpu, boxes_b_cpu = cpu_gen_boxes(shape, inp_format, r_unit)
        
        boxes_a_npu = torch.from_numpy(boxes_a_cpu).npu()
        boxes_b_npu = torch.from_numpy(boxes_b_cpu).npu()
        
        return Inputs(boxes_a_cpu, boxes_b_cpu), Inputs(boxes_a_npu, boxes_b_npu)

    def cpu_to_exec(self, cpu_inputs, attrs):
        cpu_boxes_a = cpu_inputs.boxes_a
        cpu_boxes_b = cpu_inputs.boxes_b
        cpu_res = cpu_boxes_overlap_bev(cpu_boxes_a, cpu_boxes_b, attrs)
        return cpu_res.astype(np.float32)

    def npu_to_exec(self, npu_inputs, attrs, ut_name):
        aligned, inp_format, r_unit, clockwise, mode, margin = attrs
        attrs = [self.format_dict[inp_format], self.unit_dict[r_unit], clockwise, self.mode_dict[mode], aligned, margin]
        npu_boxes_a = npu_inputs.boxes_a
        npu_boxes_b = npu_inputs.boxes_b
        
        if ut_name == "test_boxes_overlap_bev_full":
            npu_res = mx_driving._C.npu_boxes_overlap_bev(npu_boxes_a, npu_boxes_b, *attrs)
            npu_res = npu_res.cpu().float().numpy()
            return npu_res
        elif ut_name == "test_boxes_overlap_bev_bevfusion":
            npu_res1 = mx_driving.detection.boxes_overlap_bev(npu_boxes_a, npu_boxes_b)
            npu_res1 = npu_res1.cpu().float().numpy()
            npu_res2 = mx_driving.detection.npu_boxes_overlap_bev(npu_boxes_a, npu_boxes_b)
            npu_res2 = npu_res2.cpu().float().numpy()
            npu_res = mx_driving.boxes_overlap_bev(npu_boxes_a, npu_boxes_b)
            npu_res = npu_res.cpu().float().numpy()
            return npu_res, npu_res1, npu_res2
        elif ut_name == "test_boxes_overlap_bev_mmcv":
            npu_res = mx_driving._C.npu_boxes_overlap_bev(npu_boxes_a, npu_boxes_b, *attrs)
            npu_res = npu_res.cpu().float().numpy()
            return npu_res
        elif ut_name == "test_boxes_iou_bev_openpcdet":
            npu_res = mx_driving.boxes_iou_bev(npu_boxes_a, npu_boxes_b)
            npu_res = npu_res.cpu().float().numpy()
            return npu_res
        return None

    def check_precision(self, actual, expected, rtol=1e-4, atol=1e-4, msg=None):
        if not np.all(np.isclose(actual, expected, rtol=rtol, atol=atol)):
            standardMsg = f'{actual} != {expected} within relative tolerance {rtol}'
            raise AssertionError(msg or standardMsg)

    def test_boxes_overlap_bev_full(self):
        shape_list = [[12, 19], [19, 12]]
        format_list = ["xyxyr", "xywhr", "xyzxyzr", "xyzwhdr"]
        unit_list = ["radian", "degree"]
        clockwise_list = [False, True]
        mode_list = ["overlap", "iou", "iof"]
        aligned_list = [False, True]
        margin_list = [1e-5]

        cases = []
        for shape in shape_list:
            for inp_format in format_list:
                for r_unit in unit_list:
                    for clockwise in clockwise_list:
                        for mode in mode_list:
                            for aligned in aligned_list:
                                for margin in margin_list:
                                    cases.append([shape, inp_format, r_unit, clockwise, mode, aligned, margin])
        test_results = self.gen_results(cases, "test_boxes_overlap_bev_full")

        for cpu_result, npu_result in test_results:
            self.check_precision(cpu_result, npu_result, 1e-4, 1e-4)

    def test_boxes_overlap_bev_bevfusion(self):
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
        inp_format = "xyxyr"
        r_unit = "radian"
        clockwise = False
        mode = "overlap"
        aligned = False
        margin = 1e-5
        
        cases = [
            [shape, inp_format, r_unit, clockwise, mode, aligned, margin]
            for shape in shape_list
        ]
        test_results = self.gen_results(cases, "test_boxes_overlap_bev_bevfusion")

        for cpu_result, npu_results in test_results:
            for npu_result in npu_results:
                self.check_precision(cpu_result, npu_result, 1e-4, 1e-4)

    def test_boxes_overlap_bev_mmcv(self):
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
        inp_format = "xyzwhdr"
        r_unit = "radian"
        clockwise = True
        mode = "overlap"
        aligned = False
        margin = 1e-5
        
        cases = [
            [shape, inp_format, r_unit, clockwise, mode, aligned, margin]
            for shape in shape_list
        ]
        test_results = self.gen_results(cases, "test_boxes_overlap_bev_mmcv")

        for cpu_result, npu_result in test_results:
            self.check_precision(cpu_result, npu_result, 1e-4, 1e-4)

    def test_boxes_iou_bev_openpcdet(self):
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
        inp_format = "xyzwhdr"
        r_unit = "radian"
        clockwise = True
        mode = "iou"
        aligned = False
        margin = 1e-5
        
        cases = [
            [shape, inp_format, r_unit, clockwise, mode, aligned, margin]
            for shape in shape_list
        ]
        test_results = self.gen_results(cases, "test_boxes_iou_bev_openpcdet")

        for cpu_result, npu_result in test_results:
            self.check_precision(cpu_result, npu_result, 1e-4, 1e-4)
 
    def test_vertex_on_the_edge_of_another_box_boxes_iou_bev(self):
        # Test border case: A vertex of one box is on the edge of another box.
        # Test API：
        #   boxes_iou_bev（openpcdet）
        inp_format = "xyzwhdr"
        r_unit = "radian"
        clockwise = True
        mode = "iou"
        aligned = False
        margin = 1e-5
        attrs = [aligned, inp_format, r_unit, clockwise, mode, margin]

        boxes_a = torch.tensor([[-4.00859022, -1.48911846, -4.03531885, 54.45354843, 32.18401718, 72.94532776, -0.30569804]]).float()
        boxes_b = torch.tensor([[4.47819042, -0.41671070, 2.13852501, 41.19421768, 90.91581726, 97.91353607, -1.82485271],
                                [-2.35739970, 0.91445661, 2.31282878, 65.75810242, 95.59777069, 71.21990967, -2.56775522],
                                [4.77410316e+00, 8.70089699e-03, 2.35944605e+00, 6.12022209e+01, 3.45858383e+01, 4.97487717e+01, -2.95076442e+00]]).float()

        cpu_inputs = Inputs(boxes_a.cpu().numpy(), boxes_b.cpu().numpy())
        npu_inputs = Inputs(boxes_a.npu(), boxes_b.npu())

        cpu_results = self.cpu_to_exec(cpu_inputs, attrs)
        npu_results = self.npu_to_exec(npu_inputs, attrs, "test_boxes_iou_bev_openpcdet")
        self.check_precision(cpu_results, npu_results, 1e-4, 1e-4)
    
    def test_vertex_on_the_edge_of_another_box_boxes_overlap_bev(self):
        # Test border case: A vertex of one box is on the edge of another box.
        # Test API：
        #   boxes_overlap_bev（mmcv）
        inp_format = "xyzwhdr"
        r_unit = "radian"
        clockwise = True
        mode = "overlap"
        aligned = False
        margin = 1e-5
        attrs = [aligned, inp_format, r_unit, clockwise, mode, margin]

        boxes_a = torch.tensor([[-4.00859022, -1.48911846, -4.03531885, 54.45354843, 32.18401718, 72.94532776, -0.30569804]]).float()
        boxes_b = torch.tensor([[4.47819042, -0.41671070, 2.13852501, 41.19421768, 90.91581726, 97.91353607, -1.82485271],
                                [-2.35739970, 0.91445661, 2.31282878, 65.75810242, 95.59777069, 71.21990967, -2.56775522],
                                [4.77410316e+00, 8.70089699e-03, 2.35944605e+00, 6.12022209e+01, 3.45858383e+01, 4.97487717e+01, -2.95076442e+00]]).float()
        
        cpu_inputs = Inputs(boxes_a.cpu().numpy(), boxes_b.cpu().numpy())
        npu_inputs = Inputs(boxes_a.npu(), boxes_b.npu())
        
        cpu_results = self.cpu_to_exec(cpu_inputs, attrs)
        npu_results = self.npu_to_exec(npu_inputs, attrs, "test_boxes_overlap_bev_mmcv")
        self.check_precision(cpu_results, npu_results, 1e-4, 1e-4)
 
if __name__ == '__main__':
    run_tests()
