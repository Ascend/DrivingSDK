/*
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
*/
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t MAX_OBJS = 500;
constexpr int32_t BUFFER_NUM = 1;

class KernelGaussian {
public:
    __aicore__ inline KernelGaussian() {}
    __aicore__ inline void Init(GM_ADDR boxes, GM_ADDR center_int, GM_ADDR radius, GM_ADDR mask, GM_ADDR ind,
        GM_ADDR sub_xy, GM_ADDR log_box_dim, GM_ADDR sin_rot, GM_ADDR cos_rot, const GaussianTilingData* tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        numObjs = tiling_data->num_objs;
        outSizeFactor = tiling_data->out_size_factor;
        gaussianOverlap = tiling_data->gaussian_overlap;
        minRadius = tiling_data->min_radius;
        voxelSizeX = tiling_data->voxel_size_x;
        voxelSizeY = tiling_data->voxel_size_y;
        pcRangeX = tiling_data->pc_range_x;
        pcRangeY = tiling_data->pc_range_y;
        featureMapSizeX = tiling_data->feature_map_size_x;
        featureMapSizeY = tiling_data->feature_map_size_y;
        normBbox = tiling_data->norm_bbox;
        coreData = tiling_data->core_data;
        average = tiling_data->average;
        formerNum = tiling_data->former_num;

        boxesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_BOXES*>(boxes));
        centerIntGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_CENTER_INT*>(center_int));
        radiusGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_RADIUS*>(radius));
        maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_MASK*>(mask));
        indGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_IND*>(ind));
        subXYGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_SUB_XY*>(sub_xy));
        logBoxDimGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_LOG_BOX_DIM*>(log_box_dim));
        sinRotGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_SIN_ROT*>(sin_rot));
        cosRotGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_COS_ROT*>(cos_rot));
        InitBuffer();
    }

    __aicore__ inline void Process()
    {
        uint32_t core_id = GetBlockIdx();
        uint32_t tileLength = average + 1;
        uint64_t offset = core_id * tileLength;
        if (core_id >= formerNum) {
            tileLength = average;
            offset = formerNum * (tileLength + 1) + (core_id - formerNum) * tileLength;
        }
        if (tileLength == 0) {
            return;
        }
        CopyIn(tileLength, offset);
        GaussianRadius(tileLength, offset);
        Compute(tileLength, offset);
    }

    __aicore__ inline void InitBuffer()
    {
        // TBuf
        pipe.InitBuffer(indUb, coreData * sizeof(DTYPE_IND));
        pipe.InitBuffer(maskUb, AlignUp(coreData * sizeof(DTYPE_MASK), ONE_BLK_SIZE));
        pipe.InitBuffer(maskCastBuf, AlignUp(coreData * sizeof(half), ONE_BLK_SIZE));
        pipe.InitBuffer(widthMaskBuf, AlignUp(coreData * sizeof(DTYPE_MASK), 256));
        pipe.InitBuffer(lengthMaskBuf, AlignUp(coreData * sizeof(DTYPE_MASK), 256));
        pipe.InitBuffer(tempBuf, coreData * sizeof(DTYPE_BOXES));
        pipe.InitBuffer(numFpBuf, 4 * coreData * sizeof(DTYPE_BOXES));
        pipe.InitBuffer(numIntBuf, coreData * sizeof(DTYPE_RADIUS));
        pipe.InitBuffer(bUb, coreData * sizeof(DTYPE_BOXES));
        pipe.InitBuffer(cUb, coreData * sizeof(DTYPE_BOXES));
        pipe.InitBuffer(sqUb, coreData * sizeof(DTYPE_BOXES));
        pipe.InitBuffer(rUb, coreData * sizeof(DTYPE_BOXES));

        // TQue
        pipe.InitBuffer(radiusQue, BUFFER_NUM, coreData * sizeof(DTYPE_RADIUS));
        pipe.InitBuffer(boxesQue, BUFFER_NUM, 2 * coreData * sizeof(DTYPE_BOXES));
        pipe.InitBuffer(annoInQue, BUFFER_NUM, 6 * coreData * sizeof(DTYPE_BOXES));
        pipe.InitBuffer(rotQue, BUFFER_NUM, 6 * coreData * sizeof(DTYPE_BOXES));
        pipe.InitBuffer(centerQue, BUFFER_NUM, 2 * coreData * sizeof(DTYPE_CENTER_INT));
    }

    __aicore__ inline void CopyIn(uint32_t length, uint64_t offset)
    {
        uint32_t lengthAligned = AlignUp(length, B32_DATA_NUM_PER_BLOCK);
        LocalTensor<DTYPE_BOXES> boxesLocal = boxesQue.AllocTensor<DTYPE_BOXES>();
        LocalTensor<DTYPE_BOXES> annoLocal = annoInQue.AllocTensor<DTYPE_BOXES>();
        LocalTensor<DTYPE_BOXES> widthLocal = boxesLocal[lengthAligned * 0];
        LocalTensor<DTYPE_BOXES> lengthLocal = boxesLocal[lengthAligned * 1];
        LocalTensor<DTYPE_BOXES> xLocal = annoLocal[lengthAligned * 0];
        LocalTensor<DTYPE_BOXES> yLocal = annoLocal[lengthAligned * 1];
        LocalTensor<DTYPE_BOXES> rotLocal = annoLocal[lengthAligned * 2];
        LocalTensor<DTYPE_BOXES> box_dim = annoLocal[lengthAligned * 3];

        DataCopy(widthLocal, boxesGm[offset + numObjs * 3], lengthAligned);
        DataCopy(lengthLocal, boxesGm[offset + numObjs * 4], lengthAligned);
        DataCopy(xLocal, boxesGm[offset + numObjs * 0], lengthAligned);
        DataCopy(yLocal, boxesGm[offset + numObjs * 1], lengthAligned);
        DataCopy(rotLocal, boxesGm[offset + numObjs * 6], lengthAligned);
        DataCopy(box_dim, boxesGm[offset * 3 + numObjs * 3], lengthAligned * 3);

        boxesQue.EnQue(boxesLocal);
        annoInQue.EnQue(annoLocal);
    }

    __aicore__ inline void GaussianRadius(uint32_t length, uint64_t offset)
    {
        uint32_t lengthAligned = AlignUp(length, B32_DATA_NUM_PER_BLOCK);
        LocalTensor<DTYPE_BOXES> boxesLocal = boxesQue.DeQue<DTYPE_BOXES>();
        LocalTensor<DTYPE_BOXES> widthLocal = boxesLocal[0];
        LocalTensor<DTYPE_BOXES> lengthLocal = boxesLocal[lengthAligned];
        LocalTensor<DTYPE_MASK> widthMask = widthMaskBuf.Get<DTYPE_MASK>();
        LocalTensor<DTYPE_MASK> lengthMask = lengthMaskBuf.Get<DTYPE_MASK>();
        LocalTensor<DTYPE_BOXES> bLocal = bUb.Get<DTYPE_BOXES>();
        LocalTensor<DTYPE_BOXES> cLocal = cUb.Get<DTYPE_BOXES>();
        LocalTensor<DTYPE_BOXES> sqLocal = sqUb.Get<DTYPE_BOXES>();
        LocalTensor<DTYPE_BOXES> rLocal = rUb.Get<DTYPE_BOXES>();
        LocalTensor<DTYPE_BOXES> temp = tempBuf.Get<DTYPE_BOXES>();
        LocalTensor<DTYPE_BOXES> number = numFpBuf.Get<DTYPE_BOXES>();
        LocalTensor<DTYPE_RADIUS> r = numIntBuf.Get<DTYPE_RADIUS>();
        LocalTensor<DTYPE_RADIUS> radiusLocal = radiusQue.AllocTensor<DTYPE_RADIUS>();

        Duplicate(number, voxelSizeX, lengthAligned);
        Duplicate(number[lengthAligned], voxelSizeY, lengthAligned);
        Duplicate(number[lengthAligned * 2], static_cast<DTYPE_BOXES>(outSizeFactor), lengthAligned);
        Div(widthLocal, widthLocal, number, lengthAligned);
        Div(lengthLocal, lengthLocal, number[lengthAligned], lengthAligned);
        Div(widthLocal, widthLocal, number[lengthAligned * 2], lengthAligned);
        Div(lengthLocal, lengthLocal, number[lengthAligned * 2], lengthAligned);

        Add(bLocal, lengthLocal, widthLocal, lengthAligned);
        Mul(temp, widthLocal, lengthLocal, lengthAligned);
        minOverlap1 = 1 - gaussianOverlap;
        minOverlap2 = 1 + gaussianOverlap;
        Muls(cLocal, temp, minOverlap1 / minOverlap2, lengthAligned);
        Mul(sqLocal, bLocal, bLocal, lengthAligned);
        Muls(temp, cLocal, 4.0f, lengthAligned);
        Sub(temp, sqLocal, temp, lengthAligned);
        Sqrt(sqLocal, temp, lengthAligned);
        Add(temp, bLocal, sqLocal, lengthAligned);
        Duplicate(number, 2.0f, lengthAligned);
        Div(rLocal, temp, number, lengthAligned);

        Add(bLocal, lengthLocal, widthLocal, lengthAligned);
        Muls(bLocal, bLocal, 2.0f, lengthAligned);
        Mul(temp, widthLocal, lengthLocal, lengthAligned);
        Muls(cLocal, temp, minOverlap1, lengthAligned);
        Mul(sqLocal, bLocal, bLocal, lengthAligned);
        Muls(temp, cLocal, 16.0f, lengthAligned);
        Sub(temp, sqLocal, temp, lengthAligned);
        Sqrt(sqLocal, temp, lengthAligned);
        Add(temp, bLocal, sqLocal, lengthAligned);
        Div(temp, temp, number, lengthAligned);
        Min(rLocal, rLocal, temp, length);

        a = 4 * gaussianOverlap;
        minOverlap1 = -2 * gaussianOverlap;
        minOverlap2 = gaussianOverlap - 1;
        Add(bLocal, lengthLocal, widthLocal, lengthAligned);
        Muls(bLocal, bLocal, minOverlap1, lengthAligned);
        Mul(temp, widthLocal, lengthLocal, lengthAligned);
        Muls(cLocal, temp, minOverlap2, lengthAligned);
        Mul(sqLocal, bLocal, bLocal, lengthAligned);
        Muls(temp, cLocal, 4.0f * a, lengthAligned);
        Sub(temp, sqLocal, temp, lengthAligned);
        Sqrt(sqLocal, temp, lengthAligned);
        Add(temp, bLocal, sqLocal, lengthAligned);
        Div(temp, temp, number, lengthAligned);
        Min(rLocal, rLocal, temp, length);

        LocalTensor<float> radiusFp = numFpBuf.Get<float>();
        Maxs(radiusFp, rLocal, static_cast<float>(minRadius), length);
        CompareScalar(widthMask, widthLocal, 0.0f, CMPMODE::GT, AlignUp(length, 64));
        CompareScalar(lengthMask, lengthLocal, 0.0f, CMPMODE::GT, AlignUp(length, 64));
        Select(radiusFp, widthMask, radiusFp, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Select(radiusFp, lengthMask, radiusFp, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Cast(radiusLocal, radiusFp, RoundMode::CAST_FLOOR, length);
        radiusQue.EnQue(radiusLocal);

        radiusQue.DeQue<DTYPE_RADIUS>();
        DataCopyParams copyParamsRadius {1, (uint16_t)(length * sizeof(DTYPE_RADIUS)), 0, 0};
        DataCopyPad(radiusGm[offset], radiusLocal, copyParamsRadius);
        radiusQue.FreeTensor(radiusLocal);
        boxesQue.FreeTensor(boxesLocal);
    }

    __aicore__ inline void Compute(uint32_t length, uint64_t offset)
    {
        uint32_t lengthAligned = AlignUp(length, B32_DATA_NUM_PER_BLOCK);
        LocalTensor<DTYPE_BOXES> rotOut = rotQue.AllocTensor<DTYPE_BOXES>();
        LocalTensor<DTYPE_CENTER_INT> centerLocal = centerQue.AllocTensor<DTYPE_CENTER_INT>();
        LocalTensor<DTYPE_BOXES> annoLocal = annoInQue.DeQue<DTYPE_BOXES>();
        LocalTensor<DTYPE_SIN_ROT> sinRot = rotOut[0];
        LocalTensor<DTYPE_COS_ROT> cosRot = rotOut[lengthAligned];
        LocalTensor<DTYPE_CENTER_INT> xInt = centerLocal[0];
        LocalTensor<DTYPE_CENTER_INT> yInt = centerLocal[lengthAligned];
        LocalTensor<DTYPE_SUB_XY> xLocal = annoLocal[lengthAligned * 0];
        LocalTensor<DTYPE_SUB_XY> yLocal = annoLocal[lengthAligned * 1];
        LocalTensor<DTYPE_BOXES> rotLocal = annoLocal[lengthAligned * 2];
        LocalTensor<DTYPE_LOG_BOX_DIM> box_dim = annoLocal[lengthAligned * 3];
        LocalTensor<DTYPE_BOXES> number = numFpBuf.Get<DTYPE_BOXES>();
        LocalTensor<DTYPE_MASK> maskLocal = maskUb.Get<DTYPE_MASK>();
        LocalTensor<half> maskHalf = maskCastBuf.Get<half>();
        LocalTensor<DTYPE_LOG_BOX_DIM> logBoxDim = bUb.Get<DTYPE_LOG_BOX_DIM>();
        LocalTensor<DTYPE_BOXES> xIntLocal = sqUb.Get<DTYPE_BOXES>();
        LocalTensor<DTYPE_BOXES> yIntLocal = rUb.Get<DTYPE_BOXES>();
        LocalTensor<DTYPE_BOXES> ind = cUb.Get<DTYPE_BOXES>();
        LocalTensor<DTYPE_IND> indLocal = indUb.Get<DTYPE_IND>();
        LocalTensor<DTYPE_MASK> cmpWithZero = widthMaskBuf.Get<DTYPE_MASK>();
        LocalTensor<DTYPE_MASK> cmpWithGrid = lengthMaskBuf.Get<DTYPE_MASK>();

        Duplicate(number, pcRangeX, lengthAligned);
        Duplicate(number[lengthAligned], voxelSizeX, lengthAligned);
        Duplicate(number[lengthAligned * 2], pcRangeY, lengthAligned);
        Duplicate(number[lengthAligned * 3], voxelSizeY, lengthAligned);
        Sub(xLocal, xLocal, number, lengthAligned);
        Div(xLocal, xLocal, number[lengthAligned], lengthAligned);
        Sub(yLocal, yLocal, number[lengthAligned * 2], lengthAligned);
        Div(yLocal, yLocal, number[lengthAligned * 3], lengthAligned);
        Duplicate(number, static_cast<DTYPE_BOXES>(outSizeFactor), lengthAligned);
        Div(xLocal, xLocal, number, lengthAligned);
        Div(yLocal, yLocal, number, lengthAligned);

        Floor(xIntLocal, xLocal, length);
        Floor(yIntLocal, yLocal, length);
        Cast(xInt, xLocal, RoundMode::CAST_FLOOR, length);
        Cast(yInt, yLocal, RoundMode::CAST_FLOOR, length);
        Sub(xLocal, xLocal, xIntLocal, lengthAligned);
        Sub(yLocal, yLocal, yIntLocal, lengthAligned);
        centerQue.EnQue(centerLocal);

        centerQue.DeQue<DTYPE_CENTER_INT>();
        DataCopyParams copyParamsXY {1, (uint16_t)(length * sizeof(DTYPE_CENTER_INT)), 0, 0};
        DataCopyPad(centerIntGm[offset + numObjs * 0], xInt, copyParamsXY);
        DataCopyPad(centerIntGm[offset + numObjs * 1], yInt, copyParamsXY);

        // ind[k] = yInt * feature_map_size[0] + xInt
        Muls(indLocal, yInt, featureMapSizeX, lengthAligned);
        Add(indLocal, indLocal, xInt, lengthAligned);
        Cast(ind, indLocal, RoundMode::CAST_NONE, length);
        Duplicate(maskHalf, static_cast<half>(1), length);
        Select(ind, cmpWithZero, ind, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Select(ind, cmpWithGrid, ind, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Select(maskHalf, cmpWithZero, maskHalf, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Select(maskHalf, cmpWithGrid, maskHalf, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);

        // 0 <= center_int[0] < featureMapSizeX and 0 <= center_int[1] < featureMapSizeY
        CompareScalar(cmpWithZero, xIntLocal, 0.0f, CMPMODE::GE, AlignUp(length, 64));
        Select(ind, cmpWithZero, ind, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Select(maskHalf, cmpWithZero, maskHalf, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        CompareScalar(
            cmpWithGrid, xIntLocal, static_cast<DTYPE_BOXES>(featureMapSizeX), CMPMODE::LT, AlignUp(length, 64));
        Select(ind, cmpWithGrid, ind, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Select(maskHalf, cmpWithGrid, maskHalf, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        CompareScalar(cmpWithZero, yIntLocal, 0.0f, CMPMODE::GE, AlignUp(length, 64));
        Select(ind, cmpWithZero, ind, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Select(maskHalf, cmpWithZero, maskHalf, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        CompareScalar(
            cmpWithGrid, yIntLocal, static_cast<DTYPE_BOXES>(featureMapSizeY), CMPMODE::LT, AlignUp(length, 64));
        Select(ind, cmpWithGrid, ind, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Select(maskHalf, cmpWithGrid, maskHalf, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);

        Cast(indLocal, ind, RoundMode::CAST_FLOOR, length);
        Cast(maskLocal, maskHalf, RoundMode::CAST_FLOOR, length);
        Sin(sinRot, rotLocal, length);
        Cos(cosRot, rotLocal, length);
        if (normBbox) {
            Log(logBoxDim, box_dim, 3 * length);
        }
        rotQue.EnQue(rotOut);

        rotQue.DeQue<DTYPE_BOXES>();
        DataCopyParams copyParams_mask {1, (uint16_t)(length * sizeof(DTYPE_MASK)), 0, 0};
        DataCopyParams copyParams_ind {1, (uint16_t)(length * sizeof(DTYPE_IND)), 0, 0};
        DataCopyParams copyParams_ {1, (uint16_t)(length * sizeof(DTYPE_BOXES)), 0, 0};
        DataCopyParams copyParamsDim {1, (uint16_t)(3 * length * sizeof(DTYPE_LOG_BOX_DIM)), 0, 0};

        DataCopyPad(indGm[offset], indLocal, copyParams_ind);
        DataCopyPad(maskGm[offset], maskLocal, copyParams_mask);
        DataCopyPad(subXYGm[offset + MAX_OBJS * 0], xLocal, copyParams_);
        DataCopyPad(subXYGm[offset + MAX_OBJS * 1], yLocal, copyParams_);
        DataCopyPad(logBoxDimGm[offset * 3], logBoxDim, copyParamsDim);
        DataCopyPad(sinRotGm[offset], sinRot, copyParams_);
        DataCopyPad(cosRotGm[offset], cosRot, copyParams_);

        annoInQue.FreeTensor(annoLocal);
        centerQue.FreeTensor(centerLocal);
        rotQue.FreeTensor(rotOut);
    }

private:
    TPipe pipe;
    GlobalTensor<float> boxesGm, subXYGm, logBoxDimGm, sinRotGm, cosRotGm;
    GlobalTensor<int32_t> centerIntGm, radiusGm, indGm;
    GlobalTensor<uint8_t> maskGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> boxesQue, annoInQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> radiusQue, centerQue, rotQue;
    TBuf<TPosition::VECCALC> maskUb, widthMaskBuf, lengthMaskBuf, tempBuf, indUb;
    TBuf<TPosition::VECCALC> maskCastBuf, numFpBuf, numIntBuf, bUb, cUb, sqUb, rUb;

    float a;
    int32_t outSizeFactor;
    float gaussianOverlap;
    float voxelSizeX;
    float voxelSizeY;
    float pcRangeX;
    float pcRangeY;
    int32_t featureMapSizeX;
    int32_t featureMapSizeY;
    int32_t minRadius;
    uint32_t numObjs;
    uint32_t coreData;
    uint32_t average;
    uint32_t formerNum;
    bool normBbox;
    float minOverlap1;
    float minOverlap2;
};

extern "C" __global__ __aicore__ void gaussian(GM_ADDR boxes, GM_ADDR center_int, GM_ADDR radius, GM_ADDR mask,
    GM_ADDR ind, GM_ADDR sub_xy, GM_ADDR log_box_dim, GM_ADDR sin_rot, GM_ADDR cos_rot, GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGaussian op;
    op.Init(boxes, center_int, radius, mask, ind, sub_xy, log_box_dim, sin_rot, cos_rot, &tiling_data);
    op.Process();
}