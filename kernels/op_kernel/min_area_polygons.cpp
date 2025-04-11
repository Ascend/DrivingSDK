/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"
#include "cmath"

#define PI static_cast<float>(3.1415926)

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t POINT_NUM = 18;
constexpr int32_t POLYGON_NUM = 8;
constexpr int32_t ROT_DIMS = 2;
constexpr int32_t ALIGN = 16;
constexpr float AREA_MAX_VALUE = 1e12;
constexpr float COORDINATE_MAX_VALUE = 1e12;
constexpr float COORDINATE_MIN_VALUE = -1e12;
constexpr float ATAN2_DEFAULT_VALUE = 1000.0;


struct Point {
    float x, y;

    __aicore__ Point() {}

    __aicore__ Point(float _x, float _y)
    {
        x = _x;
        y = _y;
    }
};
 

template<typename T>
class KernelMinAreaPolygons {
public:
    __aicore__ inline KernelMinAreaPolygons() {}

    // 初始化
    __aicore__ inline void Init(GM_ADDR pointsets, GM_ADDR polygons,
                            const MinAreaPolygonsTilingData *__restrict tilingData)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero");
        pointsetNum = tilingData->pointsetNum;
        usedCoreNum = tilingData->usedCoreNum;
        coreTask = tilingData->coreTask;
        lastCoreTask = tilingData->lastCoreTask;
        coreId = GetBlockIdx();
        isLastCore = (coreId == (usedCoreNum - 1));
        pointsetGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *> (pointsets), pointsetNum * POINT_NUM);
        polygonGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *> (polygons), pointsetNum * POLYGON_NUM);

        pipe.InitBuffer(inQueuePointset, BUFFER_NUM, ALIGN * ROT_DIMS * sizeof(T));
        pipe.InitBuffer(outQueuePolygon, BUFFER_NUM, ALIGN * sizeof(T));
        pipe.InitBuffer(angleBuf, ALIGN * sizeof(float));
        pipe.InitBuffer(sinBuf, ALIGN * sizeof(float));
        pipe.InitBuffer(cosBuf, ALIGN * sizeof(float));
    }
    // 函数和计算逻辑的入口
    __aicore__ inline void Process()
    {
        isLastCore = (coreId == (usedCoreNum - 1));
        if (isLastCore) {
            for (int32_t i = 0; i < lastCoreTask; i++) {
                CopyIn(i);
                Compute(i);
                CopyOut(i);
            }
        } else {
            for (int32_t i = 0; i < coreTask; i++) {
                CopyIn(i);
                Compute(i);
                CopyOut(i);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t curPointset)
    {
        LocalTensor <T> pointsetLocal = inQueuePointset.AllocTensor<T> ();
        DataCopyExtParams copyParams{1, POINT_NUM * int32_t(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyPad(pointsetLocal, pointsetGm[curPointset * POINT_NUM + coreId * coreTask * POINT_NUM], copyParams, padParams);
        inQueuePointset.EnQue(pointsetLocal);
    }  // CopyIn

    __aicore__ inline void Compute(uint32_t curPointset)
    {
        LocalTensor <T> pointsetLocal = inQueuePointset.DeQue<T>();
        LocalTensor <T> polygonLocal = outQueuePolygon.AllocTensor<T>();

        float pointSet[POINT_NUM];
        for (int32_t i = 0; i < POINT_NUM; i++) {
            pointSet[i] = pointsetLocal.GetValue(i);
        }
        
        Point convex[POINT_NUM];
        uint32_t convexNum = POINT_NUM / 2;
        for (int32_t i = 0; i < convexNum; i++) {
            convex[i].x = pointSet[2 * i];
            convex[i].y = pointSet[2 * i + 1];
        }
        
        ComputeJarvisConvexHull(convex, convexNum);
        Point ps1[POINT_NUM];
        uint32_t n1 = convexNum;
        for (int32_t i = 0; i < n1; i++) {
            ps1[i].x = convex[i].x;
            ps1[i].y = convex[i].y;
        }
        ps1[n1].x = convex[0].x;
        ps1[n1].y = convex[0].y;
        
        float minbbox[5] = {0};
        MinBoundingRect(ps1, n1 + 1, minbbox);
        float angle = minbbox[0];
        float xmin = minbbox[1];
        float ymin = minbbox[2];
        float xmax = minbbox[3];
        float ymax = minbbox[4];
        float R[2][2];
        LocalTensor<float> angleLocal = angleBuf.Get<float>();
        LocalTensor<float> sinLocal = sinBuf.Get<float>();
        LocalTensor<float> cosLocal = cosBuf.Get<float>();
        angleLocal.SetValue(0, angle);
        Sin(sinLocal, angleLocal);
        Cos(cosLocal, angleLocal);
        R[0][0] = cosLocal.GetValue(0);
        R[0][1] = sinLocal.GetValue(0);
        R[1][0] = -sinLocal.GetValue(0);
        R[1][1] = cosLocal.GetValue(0);
        
        polygonLocal.SetValue(0, xmax * R[0][0] + ymin * R[1][0]);
        polygonLocal.SetValue(1, xmax * R[0][1] + ymin * R[1][1]);
        polygonLocal.SetValue(2, xmin * R[0][0] + ymin * R[1][0]);
        polygonLocal.SetValue(3, xmin * R[0][1] + ymin * R[1][1]);
        polygonLocal.SetValue(4, xmin * R[0][0] + ymax * R[1][0]);
        polygonLocal.SetValue(5, xmin * R[0][1] + ymax * R[1][1]);
        polygonLocal.SetValue(6, xmax * R[0][0] + ymax * R[1][0]);
        polygonLocal.SetValue(7, xmax * R[0][1] + ymax * R[1][1]);
        outQueuePolygon.EnQue(polygonLocal);
        inQueuePointset.FreeTensor(pointsetLocal);
    } // Compute

    __aicore__ inline void CopyOut(uint32_t curPolygon)
    {
        LocalTensor <T> polygonLocal = outQueuePolygon.DeQue<T>();
        DataCopyExtParams copyParams{1, POLYGON_NUM * sizeof(T), 0, 0, 0};
        DataCopyPad(polygonGm[curPolygon * POLYGON_NUM + coreId * coreTask * POLYGON_NUM], polygonLocal, copyParams);
        outQueuePolygon.FreeTensor(polygonLocal);
    } // CopyOut

private:
    __aicore__ inline void Swap(Point *a, Point *b)
    {
        Point temp;
        temp.x = a->x;
        temp.y = a->y;

        a->x = b->x;
        a->y = b->y;

        b->x = temp.x;
        b->y = temp.y;
    }
    __aicore__ inline float Cross(const Point o, const Point a, const Point b)
    {
        return (a.x - o.x) * (b.y - o.y) - (b.x - o.x) * (a.y - o.y);
    }
    __aicore__ inline float Dis(const Point a, const Point b)
    {
        return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    }
    __aicore__ inline void ComputeJarvisConvexHull(Point *inPoly, uint32_t &polyNum)
    {
        uint32_t inputNum = polyNum;
        Point pointMax;
        Point p;
        uint32_t maxIndex;
        uint32_t index;
        uint32_t stack[POINT_NUM];
        uint32_t top1;
        uint32_t top2;
        float sign;
        Point rightPoint[POINT_NUM];
        Point leftPoint[POINT_NUM];
        for (int32_t i = 0; i < inputNum; i++) {
            if (inPoly[i].y < inPoly[0].y ||
                inPoly[i].y == inPoly[0].y && inPoly[i].x < inPoly[0].x) {
                Point *j = &(inPoly[0]);
                Point *k = &(inPoly[i]);
                Swap(j, k);
            }
            
            if (i == 0) {
                pointMax = inPoly[0];
                maxIndex = 0;
            }

            if (inPoly[i].y > pointMax.y ||
                inPoly[i].y == pointMax.y && inPoly[i].x > pointMax.x) {
                pointMax = inPoly[i];
                maxIndex = i;
            }
        }

        if (maxIndex == 0) {
            maxIndex = 1;
            pointMax = inPoly[maxIndex];
        }

        index = 0;
        stack[0] = 0;
        top1 = 0;
        while (index != maxIndex) {
            p = pointMax;
            index = maxIndex;
            for (int32_t i = 0; i < inputNum; i++) {
                sign = Cross(inPoly[stack[top1]], inPoly[i], p);
                if ((sign > 0) || ((sign == 0) && (Dis(inPoly[stack[top1]], inPoly[i]) >
                                                Dis(inPoly[stack[top1]], p)))) {
                    p = inPoly[i];
                    index = i;
                }
            }
            top1++;
            stack[top1] = index;
        }
        
        for (int32_t i = 0; i <= top1; i++) {
            rightPoint[i] = inPoly[stack[i]];
        }
        
        index = 0;
        stack[0] = 0;
        top2 = 0;
        while (index != maxIndex) {
            p = pointMax;
            index = maxIndex;
            for (int32_t i = 1; i < inputNum; i++) {
                sign = Cross(inPoly[stack[top2]], inPoly[i], p);
                if ((sign < 0) || ((sign == 0) && (Dis(inPoly[stack[top2]], inPoly[i]) >
                                                    Dis(inPoly[stack[top2]], p)))) {
                    p = inPoly[i];
                    index = i;
                }
            }
            top2++;
            stack[top2] = index;
        }
    
        for (int32_t i = top2 - 1; i >= 0; i--) {
            leftPoint[i] = inPoly[stack[i]];
        }
        
        polyNum = top1 + top2;
        for (int32_t i = 0; i < top1 + top2; i++) {
            if (i <= top1) {
                inPoly[i] = rightPoint[i];
            } else {
                inPoly[i] = leftPoint[top2 -(i - top1)];
            }
        }
    } // ComputeJarvisConvexHull

    __aicore__ inline void MinBoundingRect(Point *ps, uint32_t points, float *minbox)
    {
        float convexPoints[2][POINT_NUM];
        for (int32_t j = 0;  j < points; j++) {
            convexPoints[0][j] = ps[j].x;
            convexPoints[1][j] = ps[j].y;
        }
        Point edges[POINT_NUM];
        float edgesAngles[POINT_NUM];
        float uniqueAngles[POINT_NUM];
        int32_t edgesNum = points - 1;
        uint32_t uniqueNum = 0;
        uint32_t uniqueFlag = 0;

        for (int32_t i = 0; i < edgesNum; i++) {
            edges[i].x = ps[i + 1].x - ps[i].x;
            edges[i].y = ps[i + 1].y - ps[i].y;
        }

        for (int32_t i = 0; i < edgesNum; i++) {
            edgesAngles[i] = MathAtan2((float)edges[i].y, (float)edges[i].x);
            if (edgesAngles[i] >= 0) {
                edgesAngles[i] = edgesAngles[i] - int32_t(edgesAngles[i] / (PI / 2)) * (PI / 2);
            } else {
                edgesAngles[i] = edgesAngles[i] - int32_t((edgesAngles[i] / (PI / 2) - 1)) * (PI / 2);
            }
        }

        uniqueAngles[0] = edgesAngles[0];
        uniqueNum += 1;
        for (int32_t i = 1; i < edgesNum; i++) {
            for (int32_t j = 0; j < uniqueNum; j++) {
                if (edgesAngles[i] == uniqueAngles[j]) {
                    uniqueFlag += 1;
                }
            }
            if (uniqueFlag == 0) {
                uniqueAngles[uniqueNum] = edgesAngles[i];
                uniqueNum += 1;
                uniqueFlag = 0;
            } else {
                uniqueFlag = 0;
            }
        }

        float minarea = AREA_MAX_VALUE;
        for (int32_t i = 0; i < uniqueNum; i++) {
            float R[2][2];
            float rot_points[2][POINT_NUM];
            LocalTensor<float> angleLocal = angleBuf.Get<float>();
            LocalTensor<float> sinLocal = sinBuf.Get<float>();
            LocalTensor<float> cosLocal = cosBuf.Get<float>();
            angleLocal.SetValue(0, uniqueAngles[i]);
            Sin(sinLocal, angleLocal);
            Cos(cosLocal, angleLocal);
            R[0][0] = cosLocal.GetValue(0);
            R[0][1] = sinLocal.GetValue(0);
            R[1][0] = -sinLocal.GetValue(0);
            R[1][1] = cosLocal.GetValue(0);

            for (int32_t m = 0; m < ROT_DIMS; m++) {
                for (int32_t n = 0; n < points; n++) {
                    float sum = 0.0;
                    for (int32_t k = 0; k < 2; k++) {
                        sum = sum + R[m][k] * convexPoints[k][n];
                    }
                    rot_points[m][n] = sum;
                }
            }
            float xmin;
            float xmax;
            float ymin;
            float ymax;
            xmin =  COORDINATE_MAX_VALUE;
            ymin = COORDINATE_MAX_VALUE;
            xmax = COORDINATE_MIN_VALUE;
            ymax = COORDINATE_MIN_VALUE;
            for (int32_t j = 0; j < points; j++) {
                if (rot_points[0][j] < xmin) {
                    xmin = rot_points[0][j];
                }
                if (rot_points[1][j] < ymin) {
                    ymin = rot_points[1][j];
                }
                if (rot_points[0][j] > xmax) {
                    xmax = rot_points[0][j];
                }
                if (rot_points[1][j] > ymax) {
                    ymax = rot_points[1][j];
                }
            }

            float area = (xmax - xmin) * (ymax - ymin);
            if (area < minarea) {
                minarea = area;
                minbox[0] = uniqueAngles[i];
                minbox[1] = xmin;
                minbox[2] = ymin;
                minbox[3] = xmax;
                minbox[4] = ymax;
            }
        }
    } // MinBoundingRect

    __aicore__ inline float MathAtan2(float a, float b)
    {
        float atan2val;
        if (b > 0) {
            atan2val = MathAtan(a / b);
        } else if ((b < 0) && (a >= 0)) {
            atan2val = MathAtan(a / b) + static_cast<float>(PI);
        } else if ((b < 0) && (a < 0)) {
            atan2val = MathAtan(a / b) - static_cast<float>(PI);
        } else if ((b == 0) && (a > 0)) {
            atan2val = static_cast<float>(PI) / 2;
        } else if ((b == 0) && (a < 0)) {
            atan2val = 0 - (static_cast<float>(PI) / 2);
        } else if ((b == 0) && (a == 0)) {
            atan2val = ATAN2_DEFAULT_VALUE;
        }
        return atan2val;
    }

    __aicore__ inline float MathAtan(const float x)
    {
        LocalTensor<float> angleLocal = angleBuf.Get<float>();
        LocalTensor<float> atanLocal = sinBuf.Get<float>();
        angleLocal.SetValue(0, x);
        Atan(atanLocal, angleLocal, 1);
        return atanLocal.GetValue(0);
    }

private:
    TPipe pipe;
    GlobalTensor<T> pointsetGm, polygonGm;
    TQue <TPosition::VECIN, BUFFER_NUM> inQueuePointset;
    TQue <TPosition::VECOUT, BUFFER_NUM> outQueuePolygon;
    TBuf <TPosition::VECCALC> angleBuf, sinBuf, cosBuf;
    uint32_t pointsetNum;
    uint32_t usedCoreNum;
    uint32_t coreTask;
    uint32_t lastCoreTask;
    uint32_t coreId;
    bool isLastCore;
}; // KernelMinAreaPolygons

extern "C" __global__ __aicore__ void min_area_polygons(GM_ADDR pointsets, GM_ADDR polygons,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KernelMinAreaPolygons<float> op;
        op.Init(pointsets, polygons, &tilingData);
        op.Process();
    }
}
 