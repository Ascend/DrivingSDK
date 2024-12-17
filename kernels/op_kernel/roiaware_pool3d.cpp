/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

using namespace AscendC;

class KernelRoiawarePool3d {
public:
    __aicore__ inline KernelRoiawarePool3d() {}
    __aicore__ inline void Init(GM_ADDR rois, GM_ADDR pts, GM_ADDR pts_feature, GM_ADDR argmax, GM_ADDR pts_idx_of_voxels, GM_ADDR pooled_features, const RoiawarePool3dTilingData *tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        coreNum = tiling_data->coreNum;
        coreRoiNums = tiling_data->coreBoxNums;
        coreRoiTail = tiling_data->coreBoxTail;
        boxNum = tiling_data->boxNum;
        ptsNum = tiling_data->ptsNum;
        channelNum = tiling_data->channelNum;
        maxPtsPerVoxel = tiling_data->maxPtsPerVoxel;
        outx = tiling_data->outx;
        outy = tiling_data->outy;
        outz = tiling_data->outz;
        mode = tiling_data->mode;

        roiDataSize = 32 / sizeof(DTYPE_ROIS);
        alignRoiNum = (roisLen + roiDataSize - 1) / roiDataSize;
        alignRoiNum = alignRoiNum * roiDataSize;

        ptsDataSize = 32 / sizeof(DTYPE_PTS);
        alignPtsNum = (ptsLen + ptsDataSize - 1) / ptsDataSize;
        alignPtsNum = alignPtsNum * ptsDataSize;

        channelDataSize = 32 / sizeof(DTYPE_POOLED_FEATURES);
        alignChannelNum = (channelNum + channelDataSize - 1) / channelDataSize;
        alignChannelNum = alignChannelNum * channelDataSize;

        maxPtsDataSize = 32 / sizeof(DTYPE_PTS_IDX_OF_VOXELS);
        alignMaxPtsNum = (maxPtsPerVoxel + maxPtsDataSize - 1) / maxPtsDataSize;
        alignMaxPtsNum = alignMaxPtsNum * maxPtsDataSize;

        uint32_t coreId = GetBlockIdx();
        if (coreId < coreRoiTail) {
            coreRoiNums += 1;
            startOffset = coreRoiNums * coreId;
        } else {
            startOffset = coreRoiNums * coreId + coreRoiTail;
        }

        roisGM.SetGlobalBuffer((__gm__ DTYPE_ROIS *)rois, static_cast<uint64_t>(boxNum) * roisLen);
        ptsGM.SetGlobalBuffer((__gm__ DTYPE_PTS *)pts, static_cast<uint64_t>(ptsNum) * ptsLen);
        ptsFeatureGM.SetGlobalBuffer((__gm__ DTYPE_PTS_FEATURE *)pts_feature, static_cast<uint64_t>(ptsNum) * channelNum);
        argmaxGM.SetGlobalBuffer((__gm__ DTYPE_ARGMAX *)argmax, static_cast<uint64_t>(boxNum) * outx * outy * outz * channelNum);
        ptsIdxOfVoxelGM.SetGlobalBuffer((__gm__ DTYPE_PTS_IDX_OF_VOXELS *)pts_idx_of_voxels, static_cast<uint64_t>(boxNum) * outx * outy * outz * maxPtsPerVoxel);
        pooledFeatureGM.SetGlobalBuffer((__gm__ DTYPE_POOLED_FEATURES *)pooled_features, static_cast<uint64_t>(boxNum) * outx * outy * outz * channelNum);
        InitBuffer();
    }

    __aicore__ inline void Process()
    {
        GetLocalTensor();
        for (uint32_t boxIdx = 0; boxIdx < coreRoiNums; boxIdx++) {
            for (uint32_t ptsIdx = 0; ptsIdx < ptsNum; ptsIdx++) {
                collect_inside_pts_for_box3d(boxIdx, ptsIdx);
            }
            
            for (uint32_t xCurIdx = 0; xCurIdx < outx; xCurIdx++) {
                for (uint32_t yCurIdx = 0; yCurIdx < outy; yCurIdx++) {
                    for (uint32_t zCurIdx = 0; zCurIdx < outz; zCurIdx++) {
                        compute_featrue(xCurIdx, yCurIdx, zCurIdx, boxIdx);
                    }
                }
            }
        }
    }

    __aicore__ inline void InitBuffer()
    {
        pipe.InitBuffer(ptsUb, alignPtsNum * sizeof(DTYPE_PTS));
        pipe.InitBuffer(roiUb, alignRoiNum * sizeof(DTYPE_ROIS));
        pipe.InitBuffer(argmaxUb, alignChannelNum * sizeof(DTYPE_ARGMAX));
        pipe.InitBuffer(idxOfVoxelUb, alignMaxPtsNum * sizeof(DTYPE_PTS_IDX_OF_VOXELS));
        pipe.InitBuffer(rzUb, 32 * sizeof(DTYPE_ROIS));
        pipe.InitBuffer(calCos, 32 * sizeof(DTYPE_ROIS));
        pipe.InitBuffer(calSin, 32 * sizeof(DTYPE_ROIS));
        pipe.InitBuffer(maxPoolFeature, alignChannelNum * sizeof(DTYPE_POOLED_FEATURES));
        pipe.InitBuffer(avgPoolFeature, alignChannelNum * sizeof(DTYPE_POOLED_FEATURES));
        pipe.InitBuffer(tmpPoolFeature, alignChannelNum * sizeof(DTYPE_POOLED_FEATURES));
        pipe.InitBuffer(avgPoolNumUb, alignChannelNum * sizeof(DTYPE_POOLED_FEATURES));
    }

    __aicore__ inline void GetLocalTensor()
    {
        ptsLocal = ptsUb.Get<DTYPE_PTS>();
        roiLocal = roiUb.Get<DTYPE_ROIS>();
        argmaxLocal = argmaxUb.Get<DTYPE_ARGMAX>();
        ptsIdVoxelLocal = idxOfVoxelUb.Get<int32_t>();
        rz = rzUb.Get<DTYPE_ROIS>();
        cosVal = calCos.Get<DTYPE_ROIS>();
        sinVal = calSin.Get<DTYPE_ROIS>();

        maxPoolLocal = maxPoolFeature.Get<DTYPE_POOLED_FEATURES>();
        avgPoolLocal = avgPoolFeature.Get<DTYPE_POOLED_FEATURES>();
        tmpPoolLocal = tmpPoolFeature.Get<DTYPE_POOLED_FEATURES>();
        avgPoolNum = avgPoolNumUb.Get<DTYPE_POOLED_FEATURES>();
    }


private:
    __aicore__ inline void collect_inside_pts_for_box3d(uint32_t boxIdx, uint32_t ptsIdx)
    {
        PipeBarrier<PIPE_ALL>();
        DataCopy(ptsLocal, ptsGM[ptsIdx * ptsLen], alignPtsNum);
        DataCopy(roiLocal, roisGM[boxIdx * roisLen + startOffset * roisLen], alignRoiNum);
        PipeBarrier<PIPE_ALL>();

        xPtsInput = ptsLocal.GetValue(xPtsDim);
        yPtsInput = ptsLocal.GetValue(yPtsDim);
        zPtsInput = ptsLocal.GetValue(zPtsDim);
        
        xRoiInput = roiLocal.GetValue(xRoiDim);
        yRoiInput = roiLocal.GetValue(yRoiDim);
        zRoiInput = roiLocal.GetValue(zRoiDim);
        xRoiSize = roiLocal.GetValue(xRoiSizeDim);
        yRoiSize = roiLocal.GetValue(yRoiSizeDim);
        zRoiSize = roiLocal.GetValue(zRoiSizeDim);
        angleRoi = roiLocal.GetValue(angleRoiDim);
        rz.SetValue(0, -angleRoi);
        int cur_flag = check_pt_in_box3d();
        if (cur_flag > 0) {
            zLocal = zPtsInput - zRoiInput;
            xRes = xRoiSize / outx;
            yRes = yRoiSize / outy;
            zRes = zRoiSize / outz;

            xIdx = int((xLocal + xRoiSize / 2) / xRes);
            yIdx = int((yLocal + yRoiSize / 2) / yRes);
            zIdx = int(zLocal / zRes);

            xIdx = min(max(xIdx, static_cast<uint32_t>(0)), outx - 1);
            yIdx = min(max(yIdx, static_cast<uint32_t>(0)), outy - 1);
            zIdx = min(max(zIdx, static_cast<uint32_t>(0)), outz - 1);

            uint64_t idOffset = (boxIdx + startOffset) * outx * outy * outz *  maxPtsPerVoxel + xIdx * outy * outz * maxPtsPerVoxel + yIdx * outz * maxPtsPerVoxel + zIdx * maxPtsPerVoxel;
            DataCopy(ptsIdVoxelLocal, ptsIdxOfVoxelGM[idOffset], alignMaxPtsNum);
            PipeBarrier<PIPE_ALL>();
            uint32_t cnt = ptsIdVoxelLocal.GetValue(0);
            if (cnt < maxPtsPerVoxel - 1) {
                ptsIdVoxelLocal.SetValue(cnt + 1, static_cast<int>(ptsIdx));
                ptsIdVoxelLocal.SetValue(0, static_cast<int>(cnt + 1));
            }
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(maxPtsPerVoxel * sizeof(DTYPE_PTS_IDX_OF_VOXELS)), 0, 0, 0};
            DataCopyPad(ptsIdxOfVoxelGM[idOffset], ptsIdVoxelLocal, copyParams);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void compute_featrue(uint32_t xCurIdx, uint32_t yCurIdx, uint32_t zCurIdx, uint32_t boxIdx)
    {
        uint64_t idOffset = (boxIdx + startOffset) * outx * outy * outz *  maxPtsPerVoxel + xCurIdx * outy * outz * maxPtsPerVoxel + yCurIdx * outz * maxPtsPerVoxel + zCurIdx * maxPtsPerVoxel;
        PipeBarrier<PIPE_ALL>();
        DataCopy(ptsIdVoxelLocal, ptsIdxOfVoxelGM[idOffset], alignMaxPtsNum);
        PipeBarrier<PIPE_ALL>();
        Duplicate(maxPoolLocal, DTYPE_POOLED_FEATURES(-1e10), alignChannelNum);
        Duplicate(avgPoolLocal, DTYPE_POOLED_FEATURES(0.0), alignChannelNum);
        Duplicate(tmpPoolLocal, DTYPE_POOLED_FEATURES(0.0), alignChannelNum);
        uint32_t ptsCurNum = ptsIdVoxelLocal.GetValue(0);
    
        for (uint32_t channelIdx = 0; channelIdx < channelNum; channelIdx++) {
            DTYPE_ARGMAX argmaxIdx = -1;
            DTYPE_POOLED_FEATURES maxVal = -1e20;
            
            for (uint32_t ptsCur = 1; ptsCur <= ptsCurNum; ptsCur++) {
                uint32_t curPtsIdx = ptsIdVoxelLocal.GetValue(ptsCur);
                PipeBarrier<PIPE_ALL>();
                DataCopy(tmpPoolLocal, ptsFeatureGM[curPtsIdx * channelNum], alignChannelNum);
                PipeBarrier<PIPE_ALL>();
                if (channelIdx == 0) {
                    Max(maxPoolLocal, maxPoolLocal, tmpPoolLocal, alignChannelNum);
                    Add(avgPoolLocal, avgPoolLocal, tmpPoolLocal, alignChannelNum);
                }
                DTYPE_POOLED_FEATURES curFeature = tmpPoolLocal.GetValue(channelIdx);
                if (curFeature > maxVal) {
                    maxVal = curFeature;
                    argmaxIdx = curPtsIdx;
                }
                argmaxLocal.SetValue(channelIdx, argmaxIdx);
            }

            if (ptsCurNum == 0 && mode == 0) {
                argmaxLocal.SetValue(channelIdx, argmaxIdx);
            }
            if (mode == 1) {
                argmaxLocal.SetValue(channelIdx, DTYPE_ARGMAX(0));
            }
        }

        uint64_t channelOffset = (boxIdx + startOffset) * outx * outy * outz * channelNum + xCurIdx * outy * outz * channelNum + yCurIdx * outz * channelNum + zCurIdx * channelNum;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(channelNum * sizeof(DTYPE_ARGMAX)), 0, 0, 0};
        DataCopyPad(argmaxGM[channelOffset], argmaxLocal, copyParams);
        PipeBarrier<PIPE_ALL>();

        if (ptsCurNum == 0) {
            Duplicate(maxPoolLocal, DTYPE_POOLED_FEATURES(0.0), alignChannelNum);
            Duplicate(avgPoolLocal, DTYPE_POOLED_FEATURES(0.0), alignChannelNum);

            DataCopyExtParams copyParamsFeature{1, static_cast<uint32_t>(channelNum * sizeof(DTYPE_POOLED_FEATURES)), 0, 0, 0};
            DataCopyPad(pooledFeatureGM[channelOffset], maxPoolLocal, copyParamsFeature);
            PipeBarrier<PIPE_ALL>();
            return;
        }

        if (mode == 0) {
            DataCopyExtParams copyParamsFeature{1, static_cast<uint32_t>(channelNum * sizeof(DTYPE_POOLED_FEATURES)), 0, 0, 0};
            DataCopyPad(pooledFeatureGM[channelOffset], maxPoolLocal, copyParamsFeature);
            PipeBarrier<PIPE_ALL>();
        } else {
            int32_t ptsCurNum = ptsIdVoxelLocal.GetValue(0);
            Duplicate(avgPoolNum, static_cast<DTYPE_POOLED_FEATURES>(ptsCurNum), alignChannelNum);
            PipeBarrier<PIPE_ALL>();
            Div(avgPoolLocal, avgPoolLocal, avgPoolNum, alignChannelNum);
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams copyParamsFeature{1, static_cast<uint32_t>(channelNum * sizeof(DTYPE_POOLED_FEATURES)), 0, 0, 0};
            DataCopyPad(pooledFeatureGM[channelOffset], avgPoolLocal, copyParamsFeature);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline bool check_pt_in_box3d()
    {
        DTYPE_ROIS cz = zRoiInput + zRoiSize / DTYPE_PTS(2);
        if (abs(zPtsInput - cz) > zRoiSize / DTYPE_PTS(2)) {
            return 0;
        }

        DTYPE_PTS shiftx, shifty;
        shiftx = xPtsInput - xRoiInput;
        shifty = yPtsInput - yRoiInput;
        Cos(cosVal[0], rz[0], 1);
        DTYPE_ROIS cosa = cosVal.GetValue(0);
        Sin(sinVal[0], rz[0], 1);
        DTYPE_ROIS sina = sinVal.GetValue(0);
        
        xLocal = shiftx * cosa + shifty * (-sina);
        yLocal = shiftx * sina + shifty * cosa;

        float in_flag = (xLocal > -xRoiSize / DTYPE_PTS(2)) & (xLocal < xRoiSize / DTYPE_PTS(2)) & (yLocal > -yRoiSize / DTYPE_PTS(2)) & (yLocal < yRoiSize / DTYPE_PTS(2));
        return in_flag;
    }

private:
    TPipe pipe;
    GlobalTensor<DTYPE_ROIS> roisGM, ptsGM, ptsFeatureGM, pooledFeatureGM;
    GlobalTensor<int32_t> argmaxGM, ptsIdxOfVoxelGM;
    
    TBuf <TPosition::VECCALC> idxOfVoxelUb;
    TBuf <TPosition::VECCALC> ptsUb;
    TBuf <TPosition::VECCALC> roiUb;
    TBuf <TPosition::VECCALC> argmaxUb;
    TBuf <TPosition::VECCALC> rzUb, calCos, calSin;
    TBuf <TPosition::VECCALC> maxPoolFeature, avgPoolFeature, tmpPoolFeature;
    TBuf <TPosition::VECCALC> avgPoolNumUb;

    LocalTensor<DTYPE_PTS_IDX_OF_VOXELS> ptsIdVoxelLocal;
    LocalTensor<DTYPE_PTS> ptsLocal;
    LocalTensor<DTYPE_ROIS> roiLocal;
    LocalTensor<DTYPE_ARGMAX> argmaxLocal;
    LocalTensor<DTYPE_ROIS> rz, cosVal, sinVal;
    LocalTensor<DTYPE_POOLED_FEATURES> maxPoolLocal, avgPoolLocal, tmpPoolLocal;
    LocalTensor<DTYPE_POOLED_FEATURES> avgPoolNum;

    uint32_t coreNum;
    uint64_t startOffset;
    uint32_t coreRoiNums, coreRoiTail;
    uint32_t boxNum, ptsNum, channelNum;
    uint32_t maxPtsPerVoxel;
    uint32_t outx, outy, outz;
    uint32_t mode;

    uint32_t roisLen = 7;
    uint32_t ptsLen = 3;

    uint32_t roiDataSize, alignRoiNum, ptsDataSize, alignPtsNum, channelDataSize, alignChannelNum, maxPtsDataSize, alignMaxPtsNum;
    uint32_t dataSize;
    uint32_t channelBatchNum = 256;
    uint32_t maxPtsBatchNum = 128;

    DTYPE_PTS xPtsInput, yPtsInput, zPtsInput;
    uint32_t xPtsDim = 0;
    uint32_t yPtsDim = 1;
    uint32_t zPtsDim = 2;
    DTYPE_ROIS xRoiInput, yRoiInput, zRoiInput, xRoiSize, yRoiSize, zRoiSize, angleRoi;
    uint32_t xRoiDim = 0;
    uint32_t yRoiDim = 1;
    uint32_t zRoiDim = 2;
    uint32_t xRoiSizeDim = 3;
    uint32_t yRoiSizeDim = 4;
    uint32_t zRoiSizeDim = 5;
    uint32_t angleRoiDim = 6;
    DTYPE_PTS xLocal, yLocal, zLocal;
    DTYPE_PTS xRes, yRes, zRes;
    uint32_t xIdx, yIdx, zIdx;
};


extern "C" __global__ __aicore__ void roiaware_pool3d(GM_ADDR rois, GM_ADDR pts, GM_ADDR pts_feature, GM_ADDR argmax, GM_ADDR pts_idx_of_voxels, GM_ADDR pooled_features, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SetSysWorkspace(workspace);
    GET_TILING_DATA(tilingData, tiling);
    const RoiawarePool3dTilingData *__restrict tilingDevice = &tilingData;

    KernelRoiawarePool3d op;
    op.Init(rois, pts, pts_feature, argmax, pts_idx_of_voxels, pooled_features, tilingDevice);
    op.Process();
}