import math
import torch
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests
from scipy.spatial import ConvexHull
from data_cache import golden_data_cache

import mx_driving
PI = 3.1415926


##数据预处理
def DataPreprocessing(P: np.array):
    ps = np.zeros((9, 2), dtype = "float32")
    for i in range(9):
        ps[i][0] = P[i * 2]
        ps[i][1] = P[i * 2 + 1]
    return ps


##求凸多边形的最小外接矩形
def minBoundingRect(ps, n_points, minbox):
    n_edges = n_points - 1
    edges = np.zeros((n_points, 2), dtype = "float32")

    for i in range(n_edges):
        edges[i][0] = ps[i + 1][0] - ps[i][0]
        edges[i][1] = ps[i + 1][1] - ps[i][1]
    edges[n_edges][0] = ps[0][0] - ps[n_edges][0]
    edges[n_edges][1] = ps[0][1] - ps[n_edges][1] 

    n_unique = 0
    unique_flag = 0
    unique_angle = []
    edges_angle = [0] * n_points
    for i in range(n_points):
        edges_angle[i] = math.atan2(edges[i][1], edges[i][0])
        if edges_angle[i] >= 0:
            edges_angle[i] = math.fmod(float(edges_angle[i]), float(PI / 2))
        else: 
            edges_angle[i] = edges_angle[i] - int((edges_angle[i] / (PI / 2) - 1)) * (PI / 2)

    unique_angle.append(edges_angle[0])
    n_unique += 1
    for i in range(1, n_points):
        for j in range(n_unique):
            if edges_angle[i] == unique_angle[j]:
                unique_flag += 1
        if unique_flag == 0:
            unique_angle.append(edges_angle[i])
            n_unique += 1
            unique_flag = 0
        else:
            unique_flag = 0

    minarea = float(1e12)
    R = np.zeros((2, 2), dtype = "float32")
    rot_points = np.zeros((2, n_points))
    for i in range(n_unique):
        R[0][0] = math.cos(unique_angle[i])
        R[0][1] = math.sin(unique_angle[i])
        R[1][0] = -math.sin(unique_angle[i])
        R[1][1] = math.cos(unique_angle[i])

        for m in range(2):
            for n in range(n_points):
                sum_rot = 0.0
                for k in range(2):
                    sum_rot = sum_rot + R[m][k] * ps[n][k]
                rot_points[m][n] = sum_rot

        xmin = float(1e12)
        for j in range(n_points):
            if math.isinf(rot_points[0][j]) or math.isnan(rot_points[0][j]):
                continue
            else:
                if rot_points[0][j] < xmin:
                    xmin = rot_points[0][j]
        ymin = float(1e12)
        for j in range(n_points):
            if math.isinf(rot_points[1][j]) or math.isnan(rot_points[1][j]):
                continue
            else:
                if rot_points[1][j] < ymin:
                    ymin = rot_points[1][j]
        xmax = float(-1e12)
        for j in range(n_points):
            if math.isinf(rot_points[0][j]) or math.isnan(rot_points[0][j]):
                continue
            else:
                if rot_points[0][j] > xmax:
                    xmax = rot_points[0][j]
        ymax = float(-1e12)
        for j in range(n_points):
            if math.isinf(rot_points[1][j]) or math.isnan(rot_points[1][j]):
                continue
            else:
                if rot_points[1][j] > ymax:
                    ymax = rot_points[1][j]
        area = float(xmax - xmin) * (ymax - ymin)

        if area < minarea:
            minarea = area
            minbox[0] = unique_angle[i]
            minbox[1] = xmin
            minbox[2] = ymin
            minbox[3] = xmax
            minbox[4] = ymax
    return minbox


def get_cpu_value(pointset):
    ps = DataPreprocessing(pointset)
    hull = ConvexHull(ps)
    p = ps[hull.vertices]
    minbbox = [0.0] * 5
    minbbox = minBoundingRect(p, p.shape[0], minbbox)
    angle = minbbox[0]
    xmin = minbbox[1]
    ymin = minbbox[2]
    xmax = minbbox[3]
    ymax = minbbox[4]

    area = float(xmax - xmin) * (ymax - ymin)
    R = np.zeros((2, 2), dtype = "float32")
    R[0][0] = math.cos(angle)
    R[0][1] = math.sin(angle)
    R[1][0] = -math.sin(angle)
    R[1][1] = math.cos(angle)

    minpoints = [0.0] * 8
    minpoints[0] = xmax * R[0][0] + ymin * R[1][0]
    minpoints[1] = xmax * R[0][1] + ymin * R[1][1]
    minpoints[2] = xmin * R[0][0] + ymin * R[1][0]
    minpoints[3] = xmin * R[0][1] + ymin * R[1][1]
    minpoints[4] = xmin * R[0][0] + ymax * R[1][0]
    minpoints[5] = xmin * R[0][1] + ymax * R[1][1]
    minpoints[6] = xmax * R[0][0] + ymax * R[1][0]
    minpoints[7] = xmax * R[0][1] + ymax * R[1][1]
    return minpoints, area


class TestMinAreaPolygons(TestCase):
    @golden_data_cache(__file__)
    def test_min_area_polygons(self):
        ## 生成数据集 Nx18
        N = 2000
        np_pointsets = np.random.uniform(-100, 100, (N, 18)).astype(np.float32)

        ## NPU计算结果
        th_pointsets = torch.from_numpy(np_pointsets).npu()
        
        npu_out = mx_driving.min_area_polygons(th_pointsets)
        npu_polygons = npu_out.cpu().numpy()
        npu_area = []
        for npu_polygon in npu_polygons:
            AB_length = math.sqrt((npu_polygon[0] - npu_polygon[2])**2 + (npu_polygon[1] - npu_polygon[3])**2)
            CD_length = math.sqrt((npu_polygon[2] - npu_polygon[4])**2 + (npu_polygon[3] - npu_polygon[5])**2)
            area = AB_length * CD_length
            npu_area.append(area)
        npu_out = np.array(npu_area)

        ## CPU计算结果
        cpu_polygons = []
        cpu_area = []
        for np_pointset in np_pointsets:
            out, area = get_cpu_value(np_pointset)
            cpu_polygons.append(out)
            cpu_area.append(area)
        cpu_out = np.array(cpu_area)
        self.assertRtolEqual(cpu_out, npu_out)
if __name__ == "__main__":
    run_tests()