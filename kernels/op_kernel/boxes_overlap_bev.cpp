/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 *
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

using namespace AscendC;

constexpr float EPS = 1e-8;
// mode flags
constexpr int32_t MODE_FLAG_OVERLAP = 0;
constexpr int32_t MODE_FLAG_IOU = 1;
constexpr int32_t MODE_FLAG_IOF = 2;
// format flags
constexpr int32_t FORMAT_FLAG_XYXYR = 0;
constexpr int32_t FORMAT_FLAG_XYWHR = 1;
constexpr int32_t FORMAT_FLAG_XYZXYZR = 2;
constexpr int32_t FORMAT_FLAG_XYZWHDR = 3;
// (x1, y1, x2, y2, angle)
constexpr uint32_t XYXYR_X1_OFFSET = 0;
constexpr uint32_t XYXYR_Y1_OFFSET = 1;
constexpr uint32_t XYXYR_X2_OFFSET = 2;
constexpr uint32_t XYXYR_Y2_OFFSET = 3;
constexpr uint32_t XYXYR_ANGLE_OFFSET = 4;
// (x_center, y_center, dx, dy, angle)
constexpr uint32_t XYWHR_XCENTER_OFFSET = 0;
constexpr uint32_t XYWHR_YCENTER_OFFSET = 1;
constexpr uint32_t XYWHR_DX_OFFSET = 2;
constexpr uint32_t XYWHR_DY_OFFSET = 3;
constexpr uint32_t XYWHR_ANGLE_OFFSET = 4;
// (x1, y1, z1, x2, y2, z2, angle)
constexpr uint32_t XYZXYZR_X1_OFFSET = 0;
constexpr uint32_t XYZXYZR_Y1_OFFSET = 1;
constexpr uint32_t XYZXYZR_Z1_OFFSET = 2;
constexpr uint32_t XYZXYZR_X2_OFFSET = 3;
constexpr uint32_t XYZXYZR_Y2_OFFSET = 4;
constexpr uint32_t XYZXYZR_Z2_OFFSET = 5;
constexpr uint32_t XYZXYZR_ANGLE_OFFSET = 6;
// (x_center, y_center, z_center, dx, dy, dz, angle)
constexpr uint32_t XYZWHDR_XCENTER_OFFSET = 0;
constexpr uint32_t XYZWHDR_YCENTER_OFFSET = 1;
constexpr uint32_t XYZWHDR_ZCENTER_OFFSET = 2;
constexpr uint32_t XYZWHDR_DX_OFFSET = 3;
constexpr uint32_t XYZWHDR_DY_OFFSET = 4;
constexpr uint32_t XYZWHDR_DZ_OFFSET = 5;
constexpr uint32_t XYZWHDR_ANGLE_OFFSET = 6;


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

template<bool clockwise, bool aligned>
class BoxesOverlapBevKernel {
public:
    __aicore__ inline BoxesOverlapBevKernel() {}
    __aicore__ inline void Init(GM_ADDR boxesA, GM_ADDR boxesB, GM_ADDR res,
                                const BoxesOverlapBevTilingData *tilingData, TPipe *tmpPipe)
    {
        pipe_ = tmpPipe;
        uint32_t curBlockIdx = GetBlockIdx();
        uint32_t blockBytes = 32;
        dataAlign_ = blockBytes / sizeof(DTYPE_RES);

        uint32_t numLargeCores = tilingData->numLargeCores;
        uint64_t numTasksPerLargeCore = tilingData->numTasksPerLargeCore;
        boxesANum_ = tilingData->boxesANum;
        boxesBNum_ = tilingData->boxesBNum;
        boxesFormatSize_ = tilingData->boxesFormatSize;
        formatFlag_ = tilingData->formatFlag;
        modeFlag_ = tilingData->modeFlag;
        margin_ = tilingData->margin;

        cpInPadExtParams_ = {false, 0, 0, 0};
        cpInPadParams_ = {1, static_cast<uint32_t>(boxesFormatSize_ * sizeof(DTYPE_RES)), 0, 0, 0};
        cpOutPadParams_ = {1, (1 * sizeof(DTYPE_RES)), 0, 0, 0};

        if (curBlockIdx < numLargeCores) {
            uint64_t numTasksCurCore = numTasksPerLargeCore;
            startOffset_ = numTasksPerLargeCore * curBlockIdx;
            endOffset_ = startOffset_ + numTasksCurCore;
        } else {
            uint64_t numTasksCurCore = numTasksPerLargeCore - 1;
            startOffset_ = numTasksPerLargeCore * numLargeCores + numTasksCurCore * (curBlockIdx - numLargeCores);
            endOffset_ = startOffset_ + numTasksCurCore;
        }

        SetGlobalBuffer(boxesA, boxesB, res);
        InitBuffer();
        GetLocalTensor();
    }

    __aicore__ inline void SetGlobalBuffer(GM_ADDR boxesA, GM_ADDR boxesB, GM_ADDR res)
    {
        boxesAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_RES *>(boxesA), static_cast<uint64_t>(boxesANum_) * boxesFormatSize_);
        boxesBGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_RES *>(boxesB), static_cast<uint64_t>(boxesBNum_) * boxesFormatSize_);
        resGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_RES *>(res), static_cast<uint64_t>(boxesANum_) * boxesBNum_);
    }

    __aicore__ inline void InitBuffer()
    {
        pipe_->InitBuffer(boxesABuf_, dataAlign_ * sizeof(DTYPE_RES));
        pipe_->InitBuffer(boxesBBuf_, dataAlign_ * sizeof(DTYPE_RES));
        pipe_->InitBuffer(resBuf_, dataAlign_ * sizeof(DTYPE_RES));
        pipe_->InitBuffer(angleBuf_, dataAlign_ * sizeof(DTYPE_RES));
        pipe_->InitBuffer(sinBuf_, dataAlign_ * sizeof(DTYPE_RES));
        pipe_->InitBuffer(cosBuf_, dataAlign_ * sizeof(DTYPE_RES));
    }

    __aicore__ inline void GetLocalTensor()
    {
        boxesALocalT_ = boxesABuf_.Get<DTYPE_RES>();
        boxesBLocalT_ = boxesBBuf_.Get<DTYPE_RES>();
        resLocalT_ = resBuf_.Get<DTYPE_RES>();
        angleLocalT_ = angleBuf_.Get<DTYPE_RES>();
        sinLocalT_ = sinBuf_.Get<DTYPE_RES>();
        cosLocalT_ = cosBuf_.Get<DTYPE_RES>();
    }

    __aicore__ inline void Process()
    {
        if (aligned) {
            for (uint64_t offsetRes = startOffset_; offsetRes < endOffset_; ++offsetRes) {
                uint64_t offsetBoxes = offsetRes * boxesFormatSize_;
                ProcessMain(offsetBoxes, offsetBoxes, offsetRes);
            }
        } else {
            for (uint64_t offsetRes = startOffset_; offsetRes < endOffset_; ++offsetRes) {
                uint64_t offsetBoxesA = static_cast<uint64_t>(offsetRes / boxesBNum_) * boxesFormatSize_;
                uint64_t offsetBoxesB = (offsetRes % boxesBNum_) * boxesFormatSize_;
                ProcessMain(offsetBoxesA, offsetBoxesB, offsetRes);
            }
        }
    }

    __aicore__ inline void ProcessMain(uint64_t offsetBoxesA, uint64_t offsetBoxesB, uint64_t offsetRes)
    {
        DataCopyPad(boxesALocalT_, boxesAGm_[offsetBoxesA], cpInPadParams_, cpInPadExtParams_);
        DataCopyPad(boxesBLocalT_, boxesBGm_[offsetBoxesB], cpInPadParams_, cpInPadExtParams_);

        DTYPE_RES res = BoxOverlap(boxesALocalT_, boxesBLocalT_);
        if (modeFlag_ == MODE_FLAG_IOU) {
            res = ComputeIoU(res);
        } else if (modeFlag_ == MODE_FLAG_IOF) {
            res = ComputeIoF(res);
        }
        resLocalT_.SetValue(0, res);
        
        DataCopyPad(resGm_[offsetRes], resLocalT_, cpOutPadParams_);
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
        if (clockwise) {
            newX = (p.x - center.x) * angleCos - (p.y - center.y) * angleSin + center.x;
            newY = (p.x - center.x) * angleSin + (p.y - center.y) * angleCos + center.y;
        } else {
            newX = (p.x - center.x) * angleCos + (p.y - center.y) * angleSin + center.x;
            newY = -(p.x - center.x) * angleSin + (p.y - center.y) * angleCos + center.y;
        }
        p.set(newX, newY);
    }

    __aicore__ inline int CheckInBox2d(const LocalTensor<float> &box, const Point &p, const float centerX, const float centerY)
    {
        Point center(centerX, centerY);
        Point rot(p.x, p.y);
        angleLocalT_.SetValue(0, -box.GetValue(4));
        Sin(sinLocalT_, angleLocalT_);
        Cos(cosLocalT_, angleLocalT_);
        float angleCos = cosLocalT_.GetValue(0);
        float angleSin = sinLocalT_.GetValue(0);
        RotateAroundCenter(center, angleCos, angleSin, rot);

        return ((rot.x > box.GetValue(0) - margin_) && (rot.x < box.GetValue(2) + margin_) &&
                (rot.y > box.GetValue(1) - margin_) && (rot.y < box.GetValue(3) + margin_));
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
        if (formatFlag_ == FORMAT_FLAG_XYXYR) {
            aX1_ = boxATensor.GetValue(XYXYR_X1_OFFSET);
            aY1_ = boxATensor.GetValue(XYXYR_Y1_OFFSET);
            aX2_ = boxATensor.GetValue(XYXYR_X2_OFFSET);
            aY2_ = boxATensor.GetValue(XYXYR_Y2_OFFSET);
            aAngle_ = boxATensor.GetValue(XYXYR_ANGLE_OFFSET);
            centerAX_ = (aX1_ + aX2_) / 2;
            centerAY_ = (aY1_ + aY2_) / 2;

            bX1_ = boxBTensor.GetValue(XYXYR_X1_OFFSET);
            bY1_ = boxBTensor.GetValue(XYXYR_Y1_OFFSET);
            bX2_ = boxBTensor.GetValue(XYXYR_X2_OFFSET);
            bY2_ = boxBTensor.GetValue(XYXYR_Y2_OFFSET);
            bAngle_ = boxBTensor.GetValue(XYXYR_ANGLE_OFFSET);
            centerBX_ = (bX1_ + bX2_) / 2;
            centerBY_ = (bY1_ + bY2_) / 2;
        } else if (formatFlag_ == FORMAT_FLAG_XYWHR) {
            centerAX_ = boxATensor.GetValue(XYWHR_XCENTER_OFFSET);
            centerAY_ = boxATensor.GetValue(XYWHR_YCENTER_OFFSET);
            aDx_ = boxATensor.GetValue(XYWHR_DX_OFFSET);
            aDy_ = boxATensor.GetValue(XYWHR_DY_OFFSET);
            aAngle_ = boxATensor.GetValue(XYWHR_ANGLE_OFFSET);
            aX1_ = centerAX_ - aDx_ / 2;
            aY1_ = centerAY_ - aDy_ / 2;
            aX2_ = centerAX_ + aDx_ / 2;
            aY2_ = centerAY_ + aDy_ / 2;

            centerBX_ = boxBTensor.GetValue(XYWHR_XCENTER_OFFSET);
            centerBY_ = boxBTensor.GetValue(XYWHR_YCENTER_OFFSET);
            bDx_ = boxBTensor.GetValue(XYWHR_DX_OFFSET);
            bDy_ = boxBTensor.GetValue(XYWHR_DY_OFFSET);
            bAngle_ = boxBTensor.GetValue(XYWHR_ANGLE_OFFSET);
            bX1_ = centerBX_ - bDx_ / 2;
            bY1_ = centerBY_ - bDy_ / 2;
            bX2_ = centerBX_ + bDx_ / 2;
            bY2_ = centerBY_ + bDy_ / 2;
        } else if (formatFlag_ == FORMAT_FLAG_XYZXYZR) {
            aX1_ = boxATensor.GetValue(XYZXYZR_X1_OFFSET);
            aY1_ = boxATensor.GetValue(XYZXYZR_Y1_OFFSET);
            aX2_ = boxATensor.GetValue(XYZXYZR_X2_OFFSET);
            aY2_ = boxATensor.GetValue(XYZXYZR_Y2_OFFSET);
            aAngle_ = boxATensor.GetValue(XYZXYZR_ANGLE_OFFSET);
            centerAX_ = (aX1_ + aX2_) / 2;
            centerAY_ = (aY1_ + aY2_) / 2;

            bX1_ = boxBTensor.GetValue(XYZXYZR_X1_OFFSET);
            bY1_ = boxBTensor.GetValue(XYZXYZR_Y1_OFFSET);
            bX2_ = boxBTensor.GetValue(XYZXYZR_X2_OFFSET);
            bY2_ = boxBTensor.GetValue(XYZXYZR_Y2_OFFSET);
            bAngle_ = boxBTensor.GetValue(XYZXYZR_ANGLE_OFFSET);
            centerBX_ = (bX1_ + bX2_) / 2;
            centerBY_ = (bY1_ + bY2_) / 2;
        } else if (formatFlag_ == FORMAT_FLAG_XYZWHDR) {
            centerAX_ = boxATensor.GetValue(XYZWHDR_XCENTER_OFFSET);
            centerAY_ = boxATensor.GetValue(XYZWHDR_YCENTER_OFFSET);
            aDx_ = boxATensor.GetValue(XYZWHDR_DX_OFFSET);
            aDy_ = boxATensor.GetValue(XYZWHDR_DY_OFFSET);
            aAngle_ = boxATensor.GetValue(XYZWHDR_ANGLE_OFFSET);
            aX1_ = centerAX_ - aDx_ / 2;
            aY1_ = centerAY_ - aDy_ / 2;
            aX2_ = centerAX_ + aDx_ / 2;
            aY2_ = centerAY_ + aDy_ / 2;

            centerBX_ = boxBTensor.GetValue(XYZWHDR_XCENTER_OFFSET);
            centerBY_ = boxBTensor.GetValue(XYZWHDR_YCENTER_OFFSET);
            bDx_ = boxBTensor.GetValue(XYZWHDR_DX_OFFSET);
            bDy_ = boxBTensor.GetValue(XYZWHDR_DY_OFFSET);
            bAngle_ = boxBTensor.GetValue(XYZWHDR_ANGLE_OFFSET);
            bX1_ = centerBX_ - bDx_ / 2;
            bY1_ = centerBY_ - bDy_ / 2;
            bX2_ = centerBX_ + bDx_ / 2;
            bY2_ = centerBY_ + bDy_ / 2;
        }

        if (modeFlag_ != MODE_FLAG_OVERLAP && formatFlag_ != FORMAT_FLAG_XYWHR && formatFlag_ != FORMAT_FLAG_XYZWHDR) {
            aDx_ = abs(aX2_ - aX1_);
            aDy_ = abs(aY2_ - aY1_);
            bDx_ = abs(bX2_ - bX1_);
            bDy_ = abs(bY2_ - bY1_);
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
            if (CheckInBox2d(boxATensor, boxBCorners[k], centerAX_, centerAY_)) {
                polyCenter = polyCenter + boxBCorners[k];
                crossPoints[count] = boxBCorners[k];
                count++;
            }
            if (CheckInBox2d(boxBTensor, boxACorners[k], centerBX_, centerBY_)) {
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

        float res = 0;
        for (int k = 0; k < count - 1; k++) {
            res += Cross(crossPoints[k] - crossPoints[0], crossPoints[k + 1] - crossPoints[0]);
        }

        return abs(res) / static_cast<float>(2.0);
    }

    __aicore__ inline float ComputeIoU(float res)
    {
        float areaA = aDx_ * aDy_;
        float areaB = bDx_ * bDy_;
        return res / max(areaA + areaB - res, EPS);
    }

    __aicore__ inline float ComputeIoF(float res)
    {
        float areaA = aDx_ * aDy_;
        return res / areaA;
    }

protected:
    TPipe *pipe_;
    GlobalTensor<DTYPE_RES> boxesAGm_, boxesBGm_, resGm_;

    TBuf<TPosition::VECCALC> boxesABuf_, boxesBBuf_, resBuf_;
    TBuf<TPosition::VECCALC> angleBuf_, sinBuf_, cosBuf_;

    LocalTensor<DTYPE_RES> resLocalT_, boxesALocalT_, boxesBLocalT_;
    LocalTensor<DTYPE_RES> angleLocalT_, sinLocalT_, cosLocalT_;

    DTYPE_RES aX1_, aX2_, aY1_, aY2_, aAngle_, centerAX_, centerAY_, aDx_, aDy_;
    DTYPE_RES bX1_, bX2_, bY1_, bY2_, bAngle_, centerBX_, centerBY_, bDx_, bDy_;

    uint64_t startOffset_, endOffset_;
    uint32_t dataAlign_;
    uint32_t boxesANum_, boxesBNum_, boxesFormatSize_;
    int32_t formatFlag_, modeFlag_;
    float margin_;

    DataCopyExtParams cpInPadParams_;
    DataCopyExtParams cpOutPadParams_;
    DataCopyPadExtParams<DTYPE_RES> cpInPadExtParams_;
};

extern "C" __global__ __aicore__ void boxes_overlap_bev(GM_ADDR boxesA, GM_ADDR boxesB, GM_ADDR res,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        BoxesOverlapBevKernel<true, true> op;
        op.Init(boxesA, boxesB, res, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        BoxesOverlapBevKernel<true, false> op;
        op.Init(boxesA, boxesB, res, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        BoxesOverlapBevKernel<false, true> op;
        op.Init(boxesA, boxesB, res, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        BoxesOverlapBevKernel<false, false> op;
        op.Init(boxesA, boxesB, res, &tilingData, &pipe);
        op.Process();
    }
}
