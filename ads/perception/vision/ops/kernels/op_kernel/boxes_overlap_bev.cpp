/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 *
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

using namespace AscendC;
constexpr float EPS = 1e-8;


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

    __aicore__ Point operator+(const Point &b) const { return Point(x + b.x, y + b.y); }

    __aicore__ Point operator-(const Point &b) const { return Point(x - b.x, y - b.y); }
};

class BoxesOverlapBevKernel {
public:
    __aicore__ inline BoxesOverlapBevKernel() {}
    __aicore__ inline void Init(GM_ADDR boxesA, GM_ADDR boxesB, GM_ADDR areaOverlap,
                                const BoxesOverlapBevTilingData *tilingData, TPipe *tmpPipe)
    {
        pipe_ = tmpPipe;
        uint32_t curBlockIdx = GetBlockIdx();
        uint32_t blockBytes = 32;
        dataAlign_ = blockBytes / sizeof(DTYPE_AREA_OVERLAP);

        uint32_t taskNum = tilingData->taskNum;
        uint32_t taskNumPerCore = tilingData->taskNumPerCore;
        boxesANum_ = tilingData->boxesANum;
        boxesBNum_ = tilingData->boxesBNum;
        outerLoopCnt_ = tilingData->outerLoopCnt;
        innerLoopCnt_ = tilingData->innerLoopCnt;
        boxesDescDimNum_ = tilingData->boxesDescDimNum;
        trans_ = tilingData->trans;
        isClockwise_ = tilingData->isClockwise;
        needIoU_ = tilingData->needIoU;

        cpInPadExtParams_ = {false, 0, 0, 0};
        cpInPadParams_ = {1, static_cast<uint32_t>(boxesDescDimNum_ * sizeof(DTYPE_AREA_OVERLAP)), 0, 0, 0};
        cpOutPadParams_ = {1, (1 * sizeof(DTYPE_AREA_OVERLAP)), 0, 0, 0};

        startOffset_ = curBlockIdx * taskNumPerCore;
        endOffset_ = (curBlockIdx + 1) * taskNumPerCore;
        if (endOffset_ > taskNum) {
            endOffset_ = taskNum;
        }

        boxesAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_AREA_OVERLAP *>(boxesA), boxesANum_ * boxesDescDimNum_);
        boxesBGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_AREA_OVERLAP *>(boxesB), boxesBNum_ * boxesDescDimNum_);
        areaOverlapGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_AREA_OVERLAP *>(areaOverlap),
                                       boxesANum_ * boxesBNum_);
    }

    __aicore__ inline void InitBuf()
    {
        pipe_->InitBuffer(boxesABuf_, dataAlign_ * sizeof(DTYPE_AREA_OVERLAP));
        pipe_->InitBuffer(boxesBBuf_, dataAlign_ * sizeof(DTYPE_AREA_OVERLAP));
        pipe_->InitBuffer(areaOverlapBuf_, dataAlign_ * sizeof(DTYPE_AREA_OVERLAP));
        pipe_->InitBuffer(angleBuf_, dataAlign_ * sizeof(DTYPE_AREA_OVERLAP));
        pipe_->InitBuffer(sinBuf_, dataAlign_ * sizeof(DTYPE_AREA_OVERLAP));
        pipe_->InitBuffer(cosBuf_, dataAlign_ * sizeof(DTYPE_AREA_OVERLAP));
    }

    __aicore__ inline void GetLocalTensor()
    {
        boxesALocalT_ = boxesABuf_.Get<DTYPE_AREA_OVERLAP>();
        boxesBLocalT_ = boxesBBuf_.Get<DTYPE_AREA_OVERLAP>();
        areaOverlapLocalT_ = areaOverlapBuf_.Get<DTYPE_AREA_OVERLAP>();
        angleLocalT_ = angleBuf_.Get<DTYPE_AREA_OVERLAP>();
        sinLocalT_ = sinBuf_.Get<DTYPE_AREA_OVERLAP>();
        cosLocalT_ = cosBuf_.Get<DTYPE_AREA_OVERLAP>();
    }

    __aicore__ inline void Process()
    {
        for (uint32_t outerId = startOffset_; outerId < endOffset_; ++outerId) {
            for (uint32_t innerId = 0; innerId < innerLoopCnt_; ++innerId) {
                uint32_t offsetBoxesA =
                    boxesANum_ > boxesBNum_ ? outerId * boxesDescDimNum_ : innerId * boxesDescDimNum_;
                uint32_t offsetBoxesB =
                    boxesANum_ > boxesBNum_ ? innerId * boxesDescDimNum_ : outerId * boxesDescDimNum_;
                uint32_t offsetAreaOverlap =
                    boxesANum_ > boxesBNum_ ? outerId * innerLoopCnt_ + innerId : innerId * outerLoopCnt_ + outerId;
                DataCopyPad(boxesALocalT_, boxesAGm_[offsetBoxesA], cpInPadParams_, cpInPadExtParams_);
                DataCopyPad(boxesBLocalT_, boxesBGm_[offsetBoxesB], cpInPadParams_, cpInPadExtParams_);
                float res = BoxOverlap(boxesALocalT_, boxesBLocalT_);
                if (needIoU_) {
                    res = ComputeIoU(res);
                }
                areaOverlapLocalT_.SetValue(0, res);
                DataCopyPad(areaOverlapGm_[offsetAreaOverlap], areaOverlapLocalT_, cpOutPadParams_);
            }
        }
    }

protected:
    __aicore__ inline float Cross(const Point &a, const Point &b) { return a.x * b.y - a.y * b.x; }

    __aicore__ inline float Cross(const Point &p1, const Point &p2, const Point &p0)
    {
        return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
    }

    __aicore__ int CheckRectCross(const Point &p1, const Point &p2, const Point &q1, const Point &q2)
    {
        int ret = (min(p1.x, p2.x) <= max(q1.x, q2.x)) && (min(q1.x, q2.x) <= max(p1.x, p2.x)) &&
                  (min(p1.y, p2.y) <= max(q1.y, q2.y)) && (min(q1.y, q2.y) <= max(p1.y, p2.y));
        return ret;
    }

    __aicore__ inline int Intersection(const Point &p1, const Point &p0, const Point &q1, const Point &q0,
                                       Point &ansPoints)
    {
        if (CheckRectCross(p0, p1, q0, q1) == 0) {
            return 0;
        }
        float s1 = Cross(q0, p1, p0);
        float s2 = Cross(p1, q1, p0);
        float s3 = Cross(p0, q1, q0);
        float s4 = Cross(q1, p1, q0);
        if (!(s1 * s2 > static_cast<float>(0.0) && s3 * s4 > static_cast<float>(0.0))) {
            return 0;
        }
        float s5 = Cross(q1, p1, p0);
        if (abs(s5 - s1) > EPS) {
            ansPoints.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
            ansPoints.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);
        } else {
            float a0 = p0.y - p1.y;
            float b0 = p1.x - p0.x;
            float c0 = p0.x * p1.y - p1.x * p0.y;
            float a1 = q0.y - q1.y;
            float b1 = q1.x - q0.x;
            float c1 = q0.x * q1.y - q1.x * q0.y;
            float D = a0 * b1 - a1 * b0;
            float adjustedD = (D == 0.0f) ? D + EPS : D;
            ansPoints.x = (b0 * c1 - b1 * c0) / D;
            ansPoints.y = (a1 * c0 - a0 * c1) / D;
        }

        return 1;
    }

    __aicore__ inline void RotateAroundCenter(const Point &center, const float angleCos, const float angleSin, Point &p)
    {
        float newX;
        float newY;
        if (isClockwise_) {
            newX = (p.x - center.x) * angleCos + (p.y - center.y) * angleSin + center.x;
            newY = -(p.x - center.x) * angleSin + (p.y - center.y) * angleCos + center.y;
        } else {
            newX = (p.x - center.x) * angleCos - (p.y - center.y) * angleSin + center.x;
            newY = (p.x - center.x) * angleSin + (p.y - center.y) * angleCos + center.y;
        }
        p.set(newX, newY);
    }

    __aicore__ inline int CheckInBox2d(const LocalTensor<float> &box, const Point &p)
    {
        const float MARGIN = 1e-5;
        float centerX = (box.GetValue(0) + box.GetValue(2)) / 2;
        float centerY = (box.GetValue(1) + box.GetValue(3)) / 2;
        Point center(centerX, centerY);
        Point rot(p.x, p.y);
        angleLocalT_.SetValue(0, -box.GetValue(4));
        Sin(sinLocalT_, angleLocalT_);
        Cos(cosLocalT_, angleLocalT_);
        float angleCos = cosLocalT_.GetValue(0);
        float angleSin = sinLocalT_.GetValue(0);
        RotateAroundCenter(center, angleCos, angleSin, rot);

        return ((rot.x > box.GetValue(0) - MARGIN) && (rot.x < box.GetValue(2) + MARGIN) &&
                (rot.y > box.GetValue(1) - MARGIN) && (rot.y < box.GetValue(3) + MARGIN));
    }

    __aicore__ inline int PointCmp(const Point &a, const Point &b, const Point &center)
    {
        float aX = a.x - center.x;
        float aY = a.y - center.y;
        float bX = b.x - center.x;
        float bY = b.y - center.y;

        if (aX >= 0 && bX < 0) {
            return true;
        } else if (aX < 0 && bX >= 0) {
            return false;
        } else {
            float slopeA = aY / aX;
            float slopeB = bY / bX;
            return slopeA > slopeB;
        }
    }

    __aicore__ inline void ParseBox(const LocalTensor<float> &boxATensor, const LocalTensor<float> &boxBTensor)
    {
        if (trans_) {
            if (boxesDescDimNum_ == 5) {
                aX2_ = boxATensor.GetValue(2);
                aY2_ = boxATensor.GetValue(3);
                aAngle_ = boxATensor.GetValue(4);

                bX2_ = boxBTensor.GetValue(2);
                bY2_ = boxBTensor.GetValue(3);
                bAngle_ = boxBTensor.GetValue(4);
            } else if (boxesDescDimNum_ == 7) {
                aX2_ = boxATensor.GetValue(3);
                aY2_ = boxATensor.GetValue(4);
                aAngle_ = boxATensor.GetValue(6);

                bX2_ = boxBTensor.GetValue(3);
                bY2_ = boxBTensor.GetValue(4);
                bAngle_ = boxBTensor.GetValue(6);
            }
            aX1_ = boxATensor.GetValue(0);
            aY1_ = boxATensor.GetValue(1);
            bX1_ = boxBTensor.GetValue(0);
            bY1_ = boxBTensor.GetValue(1);

            centerAX_ = (aX1_ + aX2_) / 2;
            centerBX_ = (bX1_ + bX2_) / 2;
            centerAY_ = (aY1_ + aY2_) / 2;
            centerBY_ = (bY1_ + bY2_) / 2;
        } else {
            if (boxesDescDimNum_ == 5) {
                aAngle_ = boxATensor.GetValue(4);
                bAngle_ = boxBTensor.GetValue(4);
                aDxHalf_ = boxATensor.GetValue(2) / 2;
                aDyHalf_ = boxATensor.GetValue(3) / 2;
                bDxHalf_ = boxBTensor.GetValue(2) / 2;
                bDyHalf_ = boxBTensor.GetValue(3) / 2;
            } else if (boxesDescDimNum_ == 7) {
                aAngle_ = boxATensor.GetValue(6);
                bAngle_ = boxBTensor.GetValue(6);
                aDxHalf_ = boxATensor.GetValue(3) / 2;
                aDyHalf_ = boxATensor.GetValue(4) / 2;
                bDxHalf_ = boxBTensor.GetValue(3) / 2;
                bDyHalf_ = boxBTensor.GetValue(4) / 2;
            }
            centerAX_ = boxATensor.GetValue(0);
            centerAY_ = boxATensor.GetValue(1);
            centerBX_ = boxBTensor.GetValue(0);
            centerBY_ = boxBTensor.GetValue(1);

            aX1_ = centerAX_ - aDxHalf_;
            aY1_ = centerAY_ - aDyHalf_;
            aX2_ = centerAX_ + aDxHalf_;
            aY2_ = centerAY_ + aDyHalf_;
            bX1_ = centerBX_ - bDxHalf_;
            bY1_ = centerBY_ - bDyHalf_;
            bX2_ = centerBX_ + bDxHalf_;
            bY2_ = centerBY_ + bDyHalf_;
        }
    }

    __aicore__ inline void updateBoxDesc(const LocalTensor<float> &boxATensor, const LocalTensor<float> &boxBTensor)
    {
        boxATensor.SetValue(0, aX1_);
        boxATensor.SetValue(1, aY1_);
        boxATensor.SetValue(2, aX2_);
        boxATensor.SetValue(3, aY2_);
        boxATensor.SetValue(4, aAngle_);

        boxBTensor.SetValue(0, bX1_);
        boxBTensor.SetValue(1, bY1_);
        boxBTensor.SetValue(2, bX2_);
        boxBTensor.SetValue(3, bY2_);
        boxBTensor.SetValue(4, bAngle_);
    }

    __aicore__ inline float BoxOverlap(const LocalTensor<float> &boxATensor, const LocalTensor<float> &boxBTensor)
    {
        ParseBox(boxATensor, boxBTensor);
        updateBoxDesc(boxATensor, boxBTensor);

        Point centerA(centerAX_, centerAY_);
        Point centerB(centerBX_, centerBY_);

        Point boxACorners[5] = {{aX1_, aY1_}, {aX2_, aY1_}, {aX2_, aY2_}, {aX1_, aY2_}, {aX1_, aY1_}};
        Point boxBCorners[5] = {{bX1_, bY1_}, {bX2_, bY1_}, {bX2_, bY2_}, {bX1_, bY2_}, {bX1_, bY1_}};

        angleLocalT_.SetValue(0, aAngle_);
        angleLocalT_.SetValue(1, bAngle_);
        Sin(sinLocalT_, angleLocalT_);
        Cos(cosLocalT_, angleLocalT_);
        float aAngleCos = cosLocalT_.GetValue(0);
        float aAngleSin = sinLocalT_.GetValue(0);
        float bAngleCos = cosLocalT_.GetValue(1);
        float bAngleSin = sinLocalT_.GetValue(1);

        for (int k = 0; k < 4; k++) {
            RotateAroundCenter(centerA, aAngleCos, aAngleSin, boxACorners[k]);
            RotateAroundCenter(centerB, bAngleCos, bAngleSin, boxBCorners[k]);
        }

        boxACorners[4] = boxACorners[0];
        boxBCorners[4] = boxBCorners[0];

        // get Intersection of lines
        Point crossPoints[16];
        Point polyCenter;
        int count = 0;
        int flag = 0;

        polyCenter.set(0, 0);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                flag = Intersection(boxACorners[i + 1], boxACorners[i], boxBCorners[j + 1], boxBCorners[j],
                                    crossPoints[count]);
                if (flag) {
                    polyCenter = polyCenter + crossPoints[count];
                    count++;
                }
            }
        }

        // check corners
        for (int k = 0; k < 4; k++) {
            if (CheckInBox2d(boxATensor, boxBCorners[k])) {
                polyCenter = polyCenter + boxBCorners[k];
                crossPoints[count] = boxBCorners[k];
                count++;
            }
            if (CheckInBox2d(boxBTensor, boxACorners[k])) {
                polyCenter = polyCenter + boxACorners[k];
                crossPoints[count] = boxACorners[k];
                count++;
            }
        }
        if (count != 0) {
            polyCenter.x /= count;
            polyCenter.y /= count;
        }

        for (size_t i = 1; i < count; ++i) {
            Point key = crossPoints[i];
            int j = i - 1;
            while (j >= 0 && PointCmp(crossPoints[j], key, polyCenter)) {
                crossPoints[j + 1] = crossPoints[j];
                --j;
            }
            crossPoints[j + 1] = key;
        }

        float areaOverlap = 0;
        for (int k = 0; k < count - 1; k++) {
            areaOverlap += Cross(crossPoints[k] - crossPoints[0], crossPoints[k + 1] - crossPoints[0]);
        }

        return abs(areaOverlap) / static_cast<float>(2.0);
    }

    __aicore__ inline float ComputeIoU(float areaOverlap)
    {
        float areaA = abs((aX2_ - aX1_) * (aY2_ - aY1_));
        float areaB = abs((bX2_ - bX1_) * (bY2_ - bY1_));
        return areaOverlap / (areaA + areaB - areaOverlap);
    }

protected:
    TPipe *pipe_;
    GlobalTensor<DTYPE_AREA_OVERLAP> boxesAGm_, boxesBGm_, areaOverlapGm_;

    TBuf<TPosition::VECCALC> boxesABuf_, boxesBBuf_, areaOverlapBuf_;
    TBuf<TPosition::VECCALC> angleBuf_, sinBuf_, cosBuf_;

    LocalTensor<DTYPE_AREA_OVERLAP> areaOverlapLocalT_, boxesALocalT_, boxesBLocalT_;
    LocalTensor<DTYPE_AREA_OVERLAP> angleLocalT_, sinLocalT_, cosLocalT_;

    DTYPE_AREA_OVERLAP aX1_, aX2_, aY1_, aY2_, aAngle_, bX1_, bX2_, bY1_, bY2_, bAngle_;
    DTYPE_AREA_OVERLAP centerAX_, centerAY_, centerBX_, centerBY_, aDxHalf_, aDyHalf_, bDxHalf_, bDyHalf_;

    uint32_t startOffset_, endOffset_;
    uint32_t dataAlign_, outerLoopCnt_, innerLoopCnt_;
    uint32_t boxesANum_, boxesBNum_, boxesDescDimNum_, 
    bool trans_, isClockwise_, needIoU_;

    DataCopyExtParams cpInPadParams_;
    DataCopyExtParams cpOutPadParams_;
    DataCopyPadExtParams<DTYPE_AREA_OVERLAP> cpInPadExtParams_;
};

extern "C" __global__ __aicore__ void boxes_overlap_bev(GM_ADDR boxesA, GM_ADDR boxesB, GM_ADDR areaOverlap,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    BoxesOverlapBevKernel op;
    op.Init(boxesA, boxesB, areaOverlap, &tilingData, &pipe);
    op.InitBuf();
    op.GetLocalTensor();
    op.Process();
}
