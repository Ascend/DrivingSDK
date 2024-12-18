/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 *
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

using namespace AscendC;

constexpr float EPS_AREA = 1e-14;
constexpr float EPS_PARALLEL = 1e-14;
constexpr float EPS_ZERO = 1e-6;
constexpr float EPS_DIST = 1e-8;
// (x_center, y_center, width, height, angle)
constexpr uint32_t ROTATED_XCENTER_OFFSET = 0;
constexpr uint32_t ROTATED_YCENTER_OFFSET = 1;
constexpr uint32_t ROTATED_WIDTH_OFFSET = 2;
constexpr uint32_t ROTATED_HEIGHT_OFFSET = 3;
constexpr uint32_t ROTATED_ANGLE_OFFSET = 4;
// (x1, y1, x2, y2, x3, y3, x4, y4)
constexpr uint32_t QUADRI_X1_OFFSET = 0;
constexpr uint32_t QUADRI_Y1_OFFSET = 1;
constexpr uint32_t QUADRI_X2_OFFSET = 2;
constexpr uint32_t QUADRI_Y2_OFFSET = 3;
constexpr uint32_t QUADRI_X3_OFFSET = 4;
constexpr uint32_t QUADRI_Y3_OFFSET = 5;
constexpr uint32_t QUADRI_X4_OFFSET = 6;
constexpr uint32_t QUADRI_Y4_OFFSET = 7;

template <typename T>
struct RotatedBox {
    T x_ctr, y_ctr, w, h, a;
};

template <typename T>
struct Point {
    T x, y;

    __aicore__ Point(const T& px = 0, const T& py = 0) : x(px), y(py) {}

    __aicore__ Point operator+(const Point& p) const
    {
        return Point(x + p.x, y + p.y);
    }

    __aicore__ Point& operator+=(const Point& p)
    {
        x += p.x;
        y += p.y;
        return *this;
    }

    __aicore__ Point operator-(const Point& p) const
    {
        return Point(x - p.x, y - p.y);
    }

    __aicore__ Point operator*(const T coeff) const
    {
        return Point(x * coeff, y * coeff);
    }
};

template<int32_t callerFlag_, int32_t modeFlag_, bool aligned_>
class BoxIouKernel {
public:
    __aicore__ inline BoxIouKernel() {}
    __aicore__ inline void Init(GM_ADDR boxesA, GM_ADDR boxesB, GM_ADDR ious,
                                const BoxIouTilingData *tilingData, TPipe *tmpPipe)
    {
        pipe_ = tmpPipe;
        uint32_t curBlockIdx = GetBlockIdx();
        uint32_t blockBytes = 32;
        dataAlign_ = blockBytes / sizeof(DTYPE_IOUS);

        uint32_t taskNum = tilingData->taskNum;
        uint32_t taskNumPerCore = tilingData->taskNumPerCore;
        boxesANum_ = tilingData->boxesANum;
        boxesBNum_ = tilingData->boxesBNum;
        outerLoopCnt_ = tilingData->outerLoopCnt;
        innerLoopCnt_ = tilingData->innerLoopCnt;
        boxesDescDimNum_ = tilingData->boxesDescDimNum;

        cpInPadExtParams_ = {false, 0, 0, 0};
        cpInPadParams_ = {1, static_cast<uint32_t>(boxesDescDimNum_ * sizeof(DTYPE_IOUS)), 0, 0, 0};
        cpOutPadParams_ = {1, (1 * sizeof(DTYPE_IOUS)), 0, 0, 0};

        startOffset_ = curBlockIdx * taskNumPerCore;
        endOffset_ = (curBlockIdx + 1) * taskNumPerCore;
        if (endOffset_ > taskNum) {
            endOffset_ = taskNum;
        }

        boxesAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_IOUS *>(boxesA), static_cast<uint64_t>(boxesANum_) * boxesDescDimNum_);
        boxesBGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_IOUS *>(boxesB), static_cast<uint64_t>(boxesBNum_) * boxesDescDimNum_);
        iousGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_IOUS *>(ious), static_cast<uint64_t>(boxesANum_) * boxesBNum_);
    }

    __aicore__ inline void InitBuf()
    {
        pipe_->InitBuffer(boxesABuf_, dataAlign_ * sizeof(DTYPE_IOUS));
        pipe_->InitBuffer(boxesBBuf_, dataAlign_ * sizeof(DTYPE_IOUS));
        pipe_->InitBuffer(iousBuf_, dataAlign_ * sizeof(DTYPE_IOUS));
        pipe_->InitBuffer(angleBuf_, dataAlign_ * sizeof(DTYPE_IOUS));
        pipe_->InitBuffer(sinBuf_, dataAlign_ * sizeof(DTYPE_IOUS));
        pipe_->InitBuffer(cosBuf_, dataAlign_ * sizeof(DTYPE_IOUS));
    }

    __aicore__ inline void GetLocalTensor()
    {
        boxesALocalT_ = boxesABuf_.Get<DTYPE_IOUS>();
        boxesBLocalT_ = boxesBBuf_.Get<DTYPE_IOUS>();
        iousLocalT_ = iousBuf_.Get<DTYPE_IOUS>();
        angleLocalT_ = angleBuf_.Get<DTYPE_IOUS>();
        sinLocalT_ = sinBuf_.Get<DTYPE_IOUS>();
        cosLocalT_ = cosBuf_.Get<DTYPE_IOUS>();
    }

    __aicore__ inline void Process()
    {
        if (aligned_) {
            for (uint64_t outerId = startOffset_; outerId < endOffset_; ++outerId) {
                uint64_t offsetBoxes = outerId * boxesDescDimNum_;
                uint64_t offsetIous = outerId;
                ProcessMain(offsetBoxes, offsetBoxes, offsetIous);
            }
        } else {
            for (uint64_t outerId = startOffset_; outerId < endOffset_; ++outerId) {
                for (uint64_t innerId = 0; innerId < innerLoopCnt_; ++innerId) {
                    uint64_t offsetBoxesA =
                        boxesANum_ > boxesBNum_ ? outerId * boxesDescDimNum_ : innerId * boxesDescDimNum_;
                    uint64_t offsetBoxesB =
                        boxesANum_ > boxesBNum_ ? innerId * boxesDescDimNum_ : outerId * boxesDescDimNum_;
                    uint64_t offsetIous =
                        boxesANum_ > boxesBNum_ ? outerId * innerLoopCnt_ + innerId : innerId * outerLoopCnt_ + outerId;
                    ProcessMain(offsetBoxesA, offsetBoxesB, offsetIous);
                }
            }
        }
    }

    __aicore__ inline void ProcessMain(uint64_t offsetBoxesA, uint64_t offsetBoxesB, uint64_t offsetIous)
    {
        DataCopyPad(boxesALocalT_, boxesAGm_[offsetBoxesA], cpInPadParams_, cpInPadExtParams_);
        DataCopyPad(boxesBLocalT_, boxesBGm_[offsetBoxesB], cpInPadParams_, cpInPadExtParams_);
        
        bool retZero = ParseBox(boxesALocalT_, boxesBLocalT_);
        if (retZero) {
            iousLocalT_.SetValue(0, static_cast<DTYPE_IOUS>(0.0));
        } else {
            DTYPE_IOUS res = ComputeBoxesIntersection();
            if (modeFlag_ == 0) {
                res = ComputeIoU(res);
            } else if (modeFlag_ == 1) {
                res = ComputeIoF(res);
            }
            iousLocalT_.SetValue(0, res);
        }
        
        DataCopyPad(iousGm_[offsetIous], iousLocalT_, cpOutPadParams_);
    }

protected:
    __aicore__ inline DTYPE_IOUS Dot(const Point<DTYPE_IOUS>& A, const Point<DTYPE_IOUS>& B)
    {
        return A.x * B.x + A.y * B.y;
    }

    __aicore__ inline DTYPE_IOUS Cross(const Point<DTYPE_IOUS>& A, const Point<DTYPE_IOUS>& B)
    {
        return A.x * B.y - B.x * A.y;
    }

    __aicore__ inline void GetRotatedVertices(const RotatedBox<DTYPE_IOUS>& box,
                                              Point<DTYPE_IOUS> (&pts)[4])
    {
        angleLocalT_.SetValue(0, box.a);
        Sin(sinLocalT_, angleLocalT_);
        Cos(cosLocalT_, angleLocalT_);
        DTYPE_IOUS cosTheta2 = cosLocalT_.GetValue(0) / 2;
        DTYPE_IOUS sinTheta2 = sinLocalT_.GetValue(0) / 2;

        pts[0].x = box.x_ctr - sinTheta2 * box.h - cosTheta2 * box.w;
        pts[0].y = box.y_ctr + cosTheta2 * box.h - sinTheta2 * box.w;
        pts[1].x = box.x_ctr + sinTheta2 * box.h - cosTheta2 * box.w;
        pts[1].y = box.y_ctr - cosTheta2 * box.h - sinTheta2 * box.w;
        pts[2].x = 2 * box.x_ctr - pts[0].x;
        pts[2].y = 2 * box.y_ctr - pts[0].y;
        pts[3].x = 2 * box.x_ctr - pts[1].x;
        pts[3].y = 2 * box.y_ctr - pts[1].y;
    }

    __aicore__ inline int GetIntersectionPoints(const Point<DTYPE_IOUS> (&pts1)[4],
                                                const Point<DTYPE_IOUS> (&pts2)[4],
                                                Point<DTYPE_IOUS> (&intersections)[24])
    {
        Point<DTYPE_IOUS> vec1[4], vec2[4];
        for (int i = 0; i < 4; i++) {
            vec1[i] = pts1[(i + 1) % 4] - pts1[i];
            vec2[i] = pts2[(i + 1) % 4] - pts2[i];
        }

        int num = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                DTYPE_IOUS det = Cross(vec2[j], vec1[i]);
                if (abs(det) <= EPS_PARALLEL) {
                    continue;
                }
                auto vec12 = pts2[j] - pts1[i];
                DTYPE_IOUS t1 = Cross(vec2[j], vec12) / det;
                DTYPE_IOUS t2 = Cross(vec1[i], vec12) / det;
                if (t1 >= 0 && t1 <= 1 && t2 >= 0 && t2 <= 1) {
                    intersections[num++] = pts1[i] + vec1[i] * t1;
                }
            }
        }

        {
            const auto& AB = vec2[0];
            const auto& DA = vec2[3];
            auto ABdotAB = Dot(AB, AB);
            auto ADdotAD = Dot(DA, DA);
            for (int i = 0; i < 4; i++) {
                auto AP = pts1[i] - pts2[0];
                auto APdotAB = Dot(AP, AB);
                auto APdotAD = -Dot(AP, DA);
                if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) &&
                    (APdotAD <= ADdotAD)) {
                    intersections[num++] = pts1[i];
                }
            }
        }

        {
            const auto& AB = vec1[0];
            const auto& DA = vec1[3];
            auto ABdotAB = Dot(AB, AB);
            auto ADdotAD = Dot(DA, DA);
            for (int i = 0; i < 4; i++) {
                auto AP = pts2[i] - pts1[0];
                auto APdotAB = Dot(AP, AB);
                auto APdotAD = -Dot(AP, DA);
                if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) &&
                    (APdotAD <= ADdotAD)) {
                    intersections[num++] = pts2[i];
                }
            }
        }

        return num;
    }

    __aicore__ inline int ConvexHullGraham(const Point<DTYPE_IOUS> (&p)[24],
                                           const int& num_in,
                                           Point<DTYPE_IOUS> (&q)[24],
                                           bool shift_to_zero = false)
    {
        int t = 0;
        for (int i = 1; i < num_in; i++) {
            if (p[i].y < p[t].y || (p[i].y == p[t].y && p[i].x < p[t].x)) {
                t = i;
            }
        }
        auto& start = p[t];

        for (int i = 0; i < num_in; i++) {
            q[i] = p[i] - start;
        }

        auto tmp = q[0];
        q[0] = q[t];
        q[t] = tmp;

        DTYPE_IOUS dist[24];
        for (int i = 0; i < num_in; i++) {
            dist[i] = Dot(q[i], q[i]);
        }

        for (int i = 1; i < num_in - 1; i++) {
            for (int j = i + 1; j < num_in; j++) {
                DTYPE_IOUS crossProduct = Cross(q[i], q[j]);
                if ((crossProduct < -EPS_ZERO) ||
                    (abs(crossProduct) < EPS_ZERO && dist[i] > dist[j])) {
                    auto q_tmp = q[i];
                    q[i] = q[j];
                    q[j] = q_tmp;
                    auto dist_tmp = dist[i];
                    dist[i] = dist[j];
                    dist[j] = dist_tmp;
                }
            }
        }

        int k;
        for (k = 1; k < num_in; k++) {
            if (dist[k] > EPS_DIST) {
                break;
            }
        }
        if (k == num_in) {
            q[0] = p[t];
            return 1;
        }
        q[1] = q[k];

        int m = 2;
        for (int i = k + 1; i < num_in; i++) {
            while (m > 1 && Cross(q[i] - q[m - 2], q[m - 1] - q[m - 2]) >= 0) {
                m--;
            }
            q[m++] = q[i];
        }

        if (!shift_to_zero) {
            for (int i = 0; i < m; i++) {
                q[i] += start;
            }
        }

        return m;
    }

    __aicore__ inline DTYPE_IOUS QuadriBoxArea(const Point<DTYPE_IOUS> (&q)[4])
    {
        DTYPE_IOUS area = 0;

        for (int i = 1; i < 3; i++) {
            area += abs(Cross(q[i] - q[0], q[i + 1] - q[0]));
        }

        return area / 2;
    }

    __aicore__ inline DTYPE_IOUS PolygonArea(const Point<DTYPE_IOUS> (&q)[24], const int& m)
    {
        if (m <= 2) {
            return 0;
        }

        DTYPE_IOUS area = 0;
        for (int i = 1; i < m - 1; i++) {
            area += abs(Cross(q[i] - q[0], q[i + 1] - q[0]));
        }

        return area / 2;
    }

    __aicore__ inline bool ParseRotatedBox(const LocalTensor<DTYPE_IOUS> &boxATensor,
                                           const LocalTensor<DTYPE_IOUS> &boxBTensor)
    {
        auto xCtrA = boxATensor.GetValue(ROTATED_XCENTER_OFFSET);
        auto yCtrA = boxATensor.GetValue(ROTATED_YCENTER_OFFSET);
        auto wA = boxATensor.GetValue(ROTATED_WIDTH_OFFSET);
        auto hA = boxATensor.GetValue(ROTATED_HEIGHT_OFFSET);
        auto angleA = boxATensor.GetValue(ROTATED_ANGLE_OFFSET);

        auto xCtrB = boxBTensor.GetValue(ROTATED_XCENTER_OFFSET);
        auto yCtrB = boxBTensor.GetValue(ROTATED_YCENTER_OFFSET);
        auto wB = boxBTensor.GetValue(ROTATED_WIDTH_OFFSET);
        auto hB = boxBTensor.GetValue(ROTATED_HEIGHT_OFFSET);
        auto angleB = boxBTensor.GetValue(ROTATED_ANGLE_OFFSET);

        RotatedBox<DTYPE_IOUS> boxA, boxB;
        auto centerShiftX = (xCtrA + xCtrB) / 2;
        auto centerShiftY = (yCtrA + yCtrB) / 2;
        
        boxA.x_ctr = xCtrA - centerShiftX;
        boxA.y_ctr = yCtrA - centerShiftY;
        boxA.w = wA;
        boxA.h = hA;
        boxA.a = angleA;

        boxB.x_ctr = xCtrB - centerShiftX;
        boxB.y_ctr = yCtrB - centerShiftY;
        boxB.w = wB;
        boxB.h = hB;
        boxB.a = angleB;

        areaA_ = boxA.w * boxA.h;
        areaB_ = boxB.w * boxB.h;
        if (areaA_ < EPS_AREA || areaB_ < EPS_AREA) {
            return true;
        }

        GetRotatedVertices(boxA, ptsA_);
        GetRotatedVertices(boxB, ptsB_);

        return false;
    }

    __aicore__ inline bool ParseQuadriBox(const LocalTensor<DTYPE_IOUS> &boxATensor,
                                          const LocalTensor<DTYPE_IOUS> &boxBTensor)
    {
        auto aX1 = boxATensor.GetValue(QUADRI_X1_OFFSET);
        auto aY1 = boxATensor.GetValue(QUADRI_Y1_OFFSET);
        auto aX2 = boxATensor.GetValue(QUADRI_X2_OFFSET);
        auto aY2 = boxATensor.GetValue(QUADRI_Y2_OFFSET);
        auto aX3 = boxATensor.GetValue(QUADRI_X3_OFFSET);
        auto aY3 = boxATensor.GetValue(QUADRI_Y3_OFFSET);
        auto aX4 = boxATensor.GetValue(QUADRI_X4_OFFSET);
        auto aY4 = boxATensor.GetValue(QUADRI_Y4_OFFSET);

        auto bX1 = boxBTensor.GetValue(QUADRI_X1_OFFSET);
        auto bY1 = boxBTensor.GetValue(QUADRI_Y1_OFFSET);
        auto bX2 = boxBTensor.GetValue(QUADRI_X2_OFFSET);
        auto bY2 = boxBTensor.GetValue(QUADRI_Y2_OFFSET);
        auto bX3 = boxBTensor.GetValue(QUADRI_X3_OFFSET);
        auto bY3 = boxBTensor.GetValue(QUADRI_Y3_OFFSET);
        auto bX4 = boxBTensor.GetValue(QUADRI_X4_OFFSET);
        auto bY4 = boxBTensor.GetValue(QUADRI_Y4_OFFSET);

        auto centerShiftX = (aX1 + aX2 + aX3 + aX4 + bX1 + bX2 + bX3 + bX4) / 8;
        auto centerShiftY = (aY1 + aY2 + aY3 + aY4 + bY1 + bY2 + bY3 + bY4) / 8;

        ptsA_[0].x = aX1 - centerShiftX;
        ptsA_[0].y = aY1 - centerShiftY;
        ptsA_[1].x = aX2 - centerShiftX;
        ptsA_[1].y = aY2 - centerShiftY;
        ptsA_[2].x = aX3 - centerShiftX;
        ptsA_[2].y = aY3 - centerShiftY;
        ptsA_[3].x = aX4 - centerShiftX;
        ptsA_[3].y = aY4 - centerShiftY;
        
        ptsB_[0].x = bX1 - centerShiftX;
        ptsB_[0].y = bY1 - centerShiftY;
        ptsB_[1].x = bX2 - centerShiftX;
        ptsB_[1].y = bY2 - centerShiftY;
        ptsB_[2].x = bX3 - centerShiftX;
        ptsB_[2].y = bY3 - centerShiftY;
        ptsB_[3].x = bX4 - centerShiftX;
        ptsB_[3].y = bY4 - centerShiftY;

        areaA_ = QuadriBoxArea(ptsA_);
        areaB_ = QuadriBoxArea(ptsB_);
        if (areaA_ < EPS_AREA || areaB_ < EPS_AREA) {
            return true;
        }

        return false;
    }

    __aicore__ inline bool ParseBox(const LocalTensor<DTYPE_IOUS> &boxATensor,
                                    const LocalTensor<DTYPE_IOUS> &boxBTensor)
    {
        if (callerFlag_ == 0) {
            return ParseRotatedBox(boxATensor, boxBTensor);
        } else if (callerFlag_ == 1) {
            return ParseQuadriBox(boxATensor, boxBTensor);
        }
        return true;
    }

    __aicore__ inline DTYPE_IOUS ComputeBoxesIntersection()
    {
        int num = GetIntersectionPoints(ptsA_, ptsB_, intersectPts_);
        if (num <= 2) {
            return static_cast<DTYPE_IOUS>(0.0);
        }
        int num_convex = ConvexHullGraham(intersectPts_, num, orderedPts_, true);
        return PolygonArea(orderedPts_, num_convex);
    }

    __aicore__ inline DTYPE_IOUS ComputeIoU(DTYPE_IOUS intersection)
    {
        return intersection / (areaA_ + areaB_ - intersection);
    }

    __aicore__ inline DTYPE_IOUS ComputeIoF(DTYPE_IOUS intersection)
    {
        return intersection / areaA_;
    }

protected:
    TPipe *pipe_;
    GlobalTensor<DTYPE_IOUS> boxesAGm_, boxesBGm_, iousGm_;

    TBuf<TPosition::VECCALC> boxesABuf_, boxesBBuf_, iousBuf_;
    TBuf<TPosition::VECCALC> angleBuf_, sinBuf_, cosBuf_;

    LocalTensor<DTYPE_IOUS> iousLocalT_, boxesALocalT_, boxesBLocalT_;
    LocalTensor<DTYPE_IOUS> angleLocalT_, sinLocalT_, cosLocalT_;

    Point<DTYPE_IOUS> ptsA_[4], ptsB_[4], intersectPts_[24], orderedPts_[24];
    DTYPE_IOUS areaA_, areaB_;

    uint32_t startOffset_, endOffset_;
    uint32_t dataAlign_, outerLoopCnt_, innerLoopCnt_;
    uint32_t boxesANum_, boxesBNum_, boxesDescDimNum_;

    DataCopyExtParams cpInPadParams_;
    DataCopyExtParams cpOutPadParams_;
    DataCopyPadExtParams<DTYPE_IOUS> cpInPadExtParams_;
};

extern "C" __global__ __aicore__ void box_iou(GM_ADDR boxesA, GM_ADDR boxesB, GM_ADDR ious,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        BoxIouKernel<0, 0, true> op;
        op.Init(boxesA, boxesB, ious, &tilingData, &pipe);
        op.InitBuf();
        op.GetLocalTensor();
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        BoxIouKernel<0, 1, true> op;
        op.Init(boxesA, boxesB, ious, &tilingData, &pipe);
        op.InitBuf();
        op.GetLocalTensor();
        op.Process();
    }else if (TILING_KEY_IS(2)) {
        BoxIouKernel<0, 0, false> op;
        op.Init(boxesA, boxesB, ious, &tilingData, &pipe);
        op.InitBuf();
        op.GetLocalTensor();
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        BoxIouKernel<0, 1, false> op;
        op.Init(boxesA, boxesB, ious, &tilingData, &pipe);
        op.InitBuf();
        op.GetLocalTensor();
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        BoxIouKernel<1, 0, true> op;
        op.Init(boxesA, boxesB, ious, &tilingData, &pipe);
        op.InitBuf();
        op.GetLocalTensor();
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        BoxIouKernel<1, 1, true> op;
        op.Init(boxesA, boxesB, ious, &tilingData, &pipe);
        op.InitBuf();
        op.GetLocalTensor();
        op.Process();
    }else if (TILING_KEY_IS(6)) {
        BoxIouKernel<1, 0, false> op;
        op.Init(boxesA, boxesB, ious, &tilingData, &pipe);
        op.InitBuf();
        op.GetLocalTensor();
        op.Process();
    } else if (TILING_KEY_IS(7)) {
        BoxIouKernel<1, 1, false> op;
        op.Init(boxesA, boxesB, ious, &tilingData, &pipe);
        op.InitBuf();
        op.GetLocalTensor();
        op.Process();
    }
}
