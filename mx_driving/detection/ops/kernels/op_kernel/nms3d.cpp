/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

#define M_PI 3.14159265358979323846 /* pi */

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr float EPS = 1e-8;
constexpr float ATAN2_DEFAULT_VALUE = 1000.0;


struct Point {
    float x, y;

    __aicore__ Point() {}

    __aicore__ Point(float _x, float _y)
    {
        x = _x;
        y = _y;
    }

    __aicore__ void set(float _x, float _y)
    {
        x = _x;
        y = _y;
    }

    __aicore__ Point operator+(const Point& b) const
    {
        return Point(x + b.x, y + b.y);
    }

    __aicore__ Point operator-(const Point& b) const
    {
        return Point(x - b.x, y - b.y);
    }
};

template<typename T>
class KernelNms3d {
public:
    __aicore__ inline KernelNms3d() {}

    __aicore__ inline void Init(GM_ADDR boxes, GM_ADDR mask, const Nms3dTilingData* __restrict tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        usedCoreNum = tiling_data->usedCoreNum;
        eachSum = tiling_data->eachSum;
        boxNum = tiling_data->boxNum;
        tailSum = tiling_data->tailSum;
        tailNum = tiling_data->tailNum;
        maskNum = tiling_data->maskNum;
        loopTime = tiling_data->loopTime;
        overlapThresh = tiling_data->overlapThresh;

        uint32_t core_id = GetBlockIdx();
        isLastCore = (core_id == (tiling_data->usedCoreNum - 1));

        boxGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(boxes), static_cast<uint64_t>(boxNum) * 7);
        maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t*>(mask), static_cast<uint64_t>(maskNum) * boxNum);

        pipe.InitBuffer(inQueueCur, BUFFER_NUM, dataAlign * sizeof(T));
        pipe.InitBuffer(inQueueBox, BUFFER_NUM, dataAlign * 7 * sizeof(T));
        pipe.InitBuffer(outQueueMask, BUFFER_NUM, dataAlign * sizeof(int16_t));
        pipe.InitBuffer(oneMask, BUFFER_NUM, dataAlign * sizeof(int16_t));

        pipe.InitBuffer(comBuf, dataAlign * sizeof(float));
        pipe.InitBuffer(p1Buf, dataAlign * sizeof(T));
        pipe.InitBuffer(p2Buf, dataAlign * sizeof(T));
        pipe.InitBuffer(q1Buf, dataAlign * sizeof(T));
        pipe.InitBuffer(q2Buf, dataAlign * sizeof(T));
        pipe.InitBuffer(angleBuf, dataAlign * sizeof(T));
        pipe.InitBuffer(sinBuf, dataAlign * sizeof(T));
        pipe.InitBuffer(cosBuf, dataAlign * sizeof(T));
        pipe.InitBuffer(pointBuf, dataAlign * sizeof(T));
        pipe.InitBuffer(min1Buf, dataAlign * sizeof(T));
        pipe.InitBuffer(min2Buf, dataAlign * sizeof(T));
        pipe.InitBuffer(max1Buf, dataAlign * sizeof(T));
        pipe.InitBuffer(max2Buf, dataAlign * sizeof(T));
        if constexpr (sizeof(T) == sizeof(half)) {
            pipe.InitBuffer(calcBuf, dataAlign * 2 * 7 * sizeof(float));
            curTemp = calcBuf.Get<float>(dataAlign * 2 * 7);
            boxTemp = curTemp[8];
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t core_id = GetBlockIdx();
        LocalTensor<int16_t> oneLocal = oneMask.AllocTensor<int16_t>();
        Duplicate(oneLocal, static_cast<int16_t>(1), dataAlign);
        for (size_t i = 0; i < boxNum; ++i) {
            for (size_t j = 0; j < loopTime; ++j) {
                uint32_t start = core_id * eachSum + dataAlign * j;
                if (i >= start + dataAlign) {
                    DataCopy(maskGm[i * maskNum + start], oneLocal, dataAlign);
                    continue;
                }
                bool is_last = (isLastCore) && (j == loopTime - 1);
                CopyIn(i, start, is_last);
                Compute(i, start, is_last);
                CopyOut(i, start);
            }
        }
        oneMask.FreeTensor(oneLocal);
    }

private:
    __aicore__ inline void CopyIn(int32_t cur_box, int32_t com_box, bool is_last)
    {
        LocalTensor<T> curLocal = inQueueCur.AllocTensor<T>();
        LocalTensor<T> boxLocal = inQueueBox.AllocTensor<T>();
        DataCopy(curLocal, boxGm[static_cast<uint64_t>(cur_box) * 7], dataAlign);
        DataCopy(boxLocal, boxGm[static_cast<uint64_t>(com_box) * 7], dataAlign * 7);
        inQueueCur.EnQue(curLocal);
        inQueueBox.EnQue(boxLocal);
    }

    __aicore__ inline void Compute(int32_t cur_box, int32_t com_box, bool is_last)
    {
        uint32_t cmpNum = is_last ? tailNum : dataAlign;
        if constexpr (sizeof(T) == sizeof(half)) {
            LocalTensor<T> curLocal = inQueueCur.DeQue<T>();
            LocalTensor<T> boxLocal = inQueueBox.DeQue<T>();
            Cast(curTemp, curLocal, RoundMode::CAST_NONE, dataAlign);
            Cast(boxTemp, boxLocal, RoundMode::CAST_NONE, 7 * dataAlign);
            inQueueCur.FreeTensor(curLocal);
            inQueueBox.FreeTensor(boxLocal);
        } else {
            curTemp = inQueueCur.DeQue<T>();
            boxTemp = inQueueBox.DeQue<T>();
        }

        PipeBarrier<PIPE_ALL>();
        LocalTensor<int16_t> outLocal = outQueueMask.AllocTensor<int16_t>();
        for (size_t i = 0; i < cmpNum; i++) {
            if (cur_box >= com_box + i) {
                outLocal.SetValue(i, 1);
                continue;
            }
            LocalTensor<float> comLocal = comBuf.Get<float>();
            for (size_t k = 0; k < 7; k++) {
                comLocal.SetValue(k, static_cast<float>(boxTemp.GetValue(i * 7 + k)));
            }
            auto flag = iou_bev(curTemp, comLocal);
            if (flag > overlapThresh) {
                outLocal.SetValue(i, 0);
            } else {
                outLocal.SetValue(i, 1);
            }
        }
        PipeBarrier<PIPE_ALL>();
        outQueueMask.EnQue<int16_t>(outLocal);
        if constexpr (sizeof(T) != sizeof(half)) {
            inQueueCur.FreeTensor(curTemp);
            inQueueBox.FreeTensor(boxTemp);
        }
    }

    __aicore__ inline void CopyOut(int32_t cur_box, int32_t com_box)
    {
        LocalTensor<int16_t> outLocal = outQueueMask.DeQue<int16_t>();
        DataCopy(maskGm[static_cast<uint64_t>(cur_box) * maskNum + static_cast<uint64_t>(com_box)], outLocal, dataAlign);
        outQueueMask.FreeTensor(outLocal);
    }

private:
    __aicore__ inline float cross(const Point& a, const Point& b)
    {
        return a.x * b.y - a.y * b.x;
    }

    __aicore__ inline float cross(const Point& p1, const Point& p2, const Point& p0)
    {
        return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
    }

    __aicore__ int check_rect_cross(const Point& p1, const Point& p2, const Point& q1, const Point& q2)
    {
        int ret = min(p1.x, p2.x) <= max(q1.x, q2.x) && min(q1.x, q2.x) <= max(p1.x, p2.x) &&
                  min(p1.y, p2.y) <= max(q1.y, q2.y) && min(q1.y, q2.y) <= max(p1.y, p2.y);
        return ret;
    }

    __aicore__ inline int check_in_box2d(const LocalTensor<float>& box, const Point& p)
    {
        const float MARGIN = 1e-2;
        float center_x = box.GetValue(0);
        float center_y = box.GetValue(1);
        LocalTensor<float> angleLocal = angleBuf.Get<float>();
        LocalTensor<float> sinLocal = sinBuf.Get<float>();
        LocalTensor<float> cosLocal = cosBuf.Get<float>();
        angleLocal.SetValue(0, -box.GetValue(6));
        Sin(sinLocal, angleLocal);
        Cos(cosLocal, angleLocal);
        float angle_cos = cosLocal.GetValue(0);
        float angle_sin = sinLocal.GetValue(0);
        float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
        float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

        return (abs(rot_x) < box.GetValue(3) / 2 + MARGIN && abs(rot_y) < box.GetValue(4) / 2 + MARGIN);
    }

    __aicore__ inline int intersection(
        const Point& p1, const Point& p0, const Point& q1, const Point& q0, Point& ans_point)
    {
        if (check_rect_cross(p0, p1, q0, q1) == 0) {
            return 0;
        }
        float s1 = cross(q0, p1, p0);
        float s2 = cross(p1, q1, p0);
        float s3 = cross(p0, q1, q0);
        float s4 = cross(q1, p1, q0);
        if (!(s1 * s2 > static_cast<float>(0.0) && s3 * s4 > static_cast<float>(0.0))) {
            return 0;
        }
        float s5 = cross(q1, p1, p0);
        if (abs(s5 - s1) > EPS) {
            float divisor = s5 - s1 == 0 ? ((s5 - s1) + EPS) : (s5 - s1);
            ans_point.x = (s5 * q0.x - s1 * q1.x) / divisor;
            ans_point.y = (s5 * q0.y - s1 * q1.y) / divisor;
        } else {
            float a0 = p0.y - p1.y;
            float b0 = p1.x - p0.x;
            float c0 = p0.x * p1.y - p1.x * p0.y;
            float a1 = q0.y - q1.y;
            float b1 = q1.x - q0.x;
            float c1 = q0.x * q1.y - q1.x * q0.y;
            float D = a0 * b1 - a1 * b0;
            float divisor = D == 0 ? D + EPS : D;
            ans_point.x = (b0 * c1 - b1 * c0) / divisor;
            ans_point.y = (a1 * c0 - a0 * c1) / divisor;
        }

        return 1;
    }

    __aicore__ inline void rotate_around_center(
        const Point& center, const float angle_cos, const float angle_sin, Point& p)
    {
        float new_x = (p.x - center.x) * angle_cos - (p.y - center.y) * angle_sin + center.x;
        float new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
        p.set(new_x, new_y);
    }

    __aicore__ inline int point_cmp(const Point& a, const Point& b, const Point& center)
    {
        return math_atan2(a.y - center.y, a.x - center.x) > math_atan2(b.y - center.y, b.x - center.x);
    }

    __aicore__ inline float math_atan2(float a, float b)
    {
        float atan2val;
        if (b > 0) {
            atan2val = math_atan(a / b);
        } else if ((b < 0) && (a >= 0)) {
            atan2val = math_atan(a / b) + static_cast<float>(M_PI);
        } else if ((b < 0) && (a < 0)) {
            atan2val = math_atan(a / b) - static_cast<float>(M_PI);
        } else if ((b == 0) && (a > 0)) {
            atan2val = static_cast<float>(M_PI) / 2;
        } else if ((b == 0) && (a < 0)) {
            atan2val = 0 - (static_cast<float>(M_PI) / 2);
        } else if ((b == 0) && (a == 0)) {
            atan2val = ATAN2_DEFAULT_VALUE;
        }
        return atan2val;
    }

    __aicore__ inline float math_atan(const float x)
    {
        LocalTensor<float> angleLocal = angleBuf.Get<float>();
        LocalTensor<float> atanLocal = sinBuf.Get<float>();
        angleLocal.SetValue(0, x);
        Atan(atanLocal, angleLocal, 1);
        return atanLocal.GetValue(0);
    }

    __aicore__ inline float box_overlap(const LocalTensor<float>& boxATensor, const LocalTensor<float>& boxBTensor)
    {
        // params box_a: [x, y, z, dx, dy, dz, heading]
        // params box_b: [x, y, z, dx, dy, dz, heading]

        float a_angle = boxATensor.GetValue(6);
        float b_angle = boxBTensor.GetValue(6);
        float a_dx_half = boxATensor.GetValue(3) / 2;
        float b_dx_half = boxBTensor.GetValue(3) / 2;
        float a_dy_half = boxATensor.GetValue(4) / 2;
        float b_dy_half = boxBTensor.GetValue(4) / 2;
        float a_x1 = boxATensor.GetValue(0) - a_dx_half;
        float a_y1 = boxATensor.GetValue(1) - a_dy_half;
        float a_x2 = boxATensor.GetValue(0) + a_dx_half;
        float a_y2 = boxATensor.GetValue(1) + a_dy_half;
        float b_x1 = boxBTensor.GetValue(0) - b_dx_half;
        float b_y1 = boxBTensor.GetValue(1) - b_dy_half;
        float b_x2 = boxBTensor.GetValue(0) + b_dx_half;
        float b_y2 = boxBTensor.GetValue(1) + b_dy_half;

        Point center_a(boxATensor.GetValue(0), boxATensor.GetValue(1));
        Point center_b(boxBTensor.GetValue(0), boxBTensor.GetValue(1));

        Point box_a_corners[5] = {{a_x1, a_y1}, {a_x2, a_y1}, {a_x2, a_y2}, {a_x1, a_y2}, {a_x1, a_y1}};
        Point box_b_corners[5] = {{b_x1, b_y1}, {b_x2, b_y1}, {b_x2, b_y2}, {b_x1, b_y2}, {b_x1, b_y1}};

        // get oriented corners
        LocalTensor<float> angleLocal = angleBuf.Get<float>();
        LocalTensor<float> sinLocal = sinBuf.Get<float>();
        LocalTensor<float> cosLocal = cosBuf.Get<float>();
        angleLocal.SetValue(0, a_angle);
        angleLocal.SetValue(1, b_angle);
        Sin(sinLocal, angleLocal);
        Cos(cosLocal, angleLocal);
        float a_angle_cos = cosLocal.GetValue(0);
        float a_angle_sin = sinLocal.GetValue(0);
        float b_angle_cos = cosLocal.GetValue(1);
        float b_angle_sin = sinLocal.GetValue(1);

        for (int k = 0; k < 4; k++) {
            rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
            rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
        }

        box_a_corners[4] = box_a_corners[0];
        box_b_corners[4] = box_b_corners[0];

        // get intersection of lines
        Point cross_points[16];
        Point poly_center;
        int count = 0;
        int flag = 0;

        poly_center.set(0, 0);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                flag = intersection(box_a_corners[i + 1], box_a_corners[i], box_b_corners[j + 1], box_b_corners[j],
                    cross_points[count]);
                if (flag) {
                    poly_center = poly_center + cross_points[count];
                    count++;
                }
            }
        }

        // check corners
        for (int k = 0; k < 4; k++) {
            if (check_in_box2d(boxATensor, box_b_corners[k])) {
                poly_center = poly_center + box_b_corners[k];
                cross_points[count] = box_b_corners[k];
                count++;
            }
            if (check_in_box2d(boxBTensor, box_a_corners[k])) {
                poly_center = poly_center + box_a_corners[k];
                cross_points[count] = box_a_corners[k];
                count++;
            }
        }
        if (count != 0) {
            poly_center.x /= count;
            poly_center.y /= count;
        }

        for (size_t i = 1; i < count; ++i) {
            Point key = cross_points[i];
            int j = i - 1;
            while (j >= 0 && point_cmp(cross_points[j], key, poly_center)) {
                cross_points[j + 1] = cross_points[j];
                --j;
            }
            cross_points[j + 1] = key;
        }

        float cross_area = 0;
        for (int k = 0; k < count - 1; k++) {
            cross_area += cross(cross_points[k] - cross_points[0], cross_points[k + 1] - cross_points[0]);
        }

        return abs(cross_area) / static_cast<float>(2.0);
    }

    __aicore__ inline float iou_bev(const LocalTensor<float>& boxATensor, const LocalTensor<float>& boxBTensor)
    {
        // params box_a: [x, y, z, dx, dy, dz, heading]
        // params box_b: [x, y, z, dx, dy, dz, heading]
        float sa = boxATensor.GetValue(3) * boxATensor.GetValue(4);
        float sb = boxBTensor.GetValue(3) * boxBTensor.GetValue(4);
        float s_overlap = box_overlap(boxATensor, boxBTensor);
        return s_overlap / max(sa + sb - s_overlap, EPS);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueCur, inQueueBox;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueMask, oneMask;
    TBuf<TPosition::VECCALC> calcBuf;
    TBuf<TPosition::VECCALC> comBuf;

    TBuf<TPosition::VECCALC> p1Buf, p2Buf, q1Buf, q2Buf;
    TBuf<TPosition::VECCALC> angleBuf, sinBuf, cosBuf, pointBuf;
    TBuf<TPosition::VECCALC> min1Buf, min2Buf, max1Buf, max2Buf;

    GlobalTensor<T> boxGm;
    GlobalTensor<int16_t> maskGm;
    LocalTensor<float> curTemp, boxTemp;
    uint32_t usedCoreNum;
    uint32_t loopTime;
    uint32_t eachSum;
    uint32_t boxNum;
    uint32_t tailSum;
    uint32_t tailNum;
    uint32_t maskNum;
    uint32_t dataAlign = 16;
    float overlapThresh;
    bool isLastCore;
};

extern "C" __global__ __aicore__ void nms3d(GM_ADDR boxes, GM_ADDR mask, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    const Nms3dTilingData* __restrict tilingDevice = &tilingData;
    if (TILING_KEY_IS(1)) {
        KernelNms3d<float> op;
        op.Init(boxes, mask, tilingDevice);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelNms3d<half> op;
        op.Init(boxes, mask, tilingDevice);
        op.Process();
    }
}