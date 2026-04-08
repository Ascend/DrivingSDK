/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/

#include "kernel_operator.h"

using namespace AscendC;

namespace {
    constexpr int32_t INT_MAX = 2147483647;
    constexpr int32_t INT_MIN = -2147483648;
    constexpr int32_t BUFFER_NUM = 1;
    constexpr int32_t INPUT_NUM = 3;
    constexpr int32_t OUTPUT_NUM = 2;
    constexpr int32_t GRAD_INPUT_INDEX = 0;
    constexpr int32_t X_INPUT_INDEX = 1;
    constexpr int32_t GRID_INPUT_INDEX = 2;
    constexpr int32_t DX_INPUT_INDEX = 3;
    constexpr int32_t DGRID_INPUT_INDEX = 4;
    constexpr int32_t GM_PARAMS_SIZE = 5;
    constexpr int32_t DX_OUTPUT_INDEX = 0;
    constexpr int32_t DGRID_OUTPUT_INDEX = 1;
    constexpr int32_t BLOCK_BYTES = 32;
    constexpr int32_t UINT8_BITS = 8;
    constexpr int32_t ELE_NUM_PER_REPEAT = 96;
    constexpr int32_t GATHER_MASK_NUM = 96;
    constexpr int32_t REPEAT_STRIDE_0 = 12;
    constexpr int32_t REPEAT_STRIDE_1 = 0;
    constexpr int32_t FLOAT_BYTES = 4;
    constexpr int32_t ALIGN_256_BYTES = 256;

    template <typename T>
    class KernelGridSampler3dGrad {
    public:
        __aicore__ inline KernelGridSampler3dGrad(){};
        __aicore__ inline void Init(GridSampler3dGradV1TilingData* tilingData, GM_ADDR inputTensors[GM_PARAMS_SIZE + 1], TPipe* pipe);
        __aicore__ inline void Process();

    private:
        __aicore__ inline void InitBilinearBuffer(TPipe* pipe);
        __aicore__ inline void InitBilinearLocalTensor();
        __aicore__ inline void ComputeWeight(LocalTensor<T> dst, LocalTensor<T> xCoorTensor1, LocalTensor<T> xCoorTensor2,
                                            LocalTensor<T> yCoorTensor1, LocalTensor<T> yCoorTensor2,
                                            LocalTensor<T> zCoorTensor1, LocalTensor<T> zCoorTensor2,
                                            const int32_t calCount);
        __aicore__ inline void DupValue();
        __aicore__ inline void ComputeSourceIndexSetGrad(LocalTensor<T> dataTensor, LocalTensor<T> gradTensor, const T size,
                                                        const int32_t calCount);

        __aicore__ inline void WithinBounds3d(LocalTensor<T> dst, LocalTensor<T> izT, LocalTensor<T> iyT, LocalTensor<T> ixT,
                                                LocalTensor<T> weight, const int32_t calCount);
        __aicore__ inline void ComputeIndex(LocalTensor<int32_t> dstIndex, LocalTensor<int32_t> dstIndex2,
                                            LocalTensor<int32_t> zCoor, LocalTensor<int32_t> yCoor,
                                            LocalTensor<int32_t> xCoor, const int32_t calCount);
        __aicore__ inline void ComputeAfterTransposeGridGrad(LocalTensor<int32_t> srcIndex, LocalTensor<T> zCoor1,
                                                            LocalTensor<T> zCoor2, LocalTensor<T> yCoor1,
                                                            LocalTensor<T> yCoor2, LocalTensor<T> xCoor1,
                                                            LocalTensor<T> xCoor2, LocalTensor<T> gOutLocalTensor,
                                                            LocalTensor<T> selTensor, const int32_t coorIndex,
                                                            const int32_t batchIdx, int32_t xTag, int32_t yTag,
                                                            int32_t zTag);
        __aicore__ inline void ComputeAfterTransposeXGrad(LocalTensor<int32_t> srcIndex, LocalTensor<T> weight,
                                                            const int32_t coorIndex, const int64_t ncOffset,
                                                            LocalTensor<T> gOutLocalTensor);

        __aicore__ inline void InitComputeTensor();
        __aicore__ inline void ComputeBilinear(int32_t actualComputeCount, const int64_t curGridPointIndex);
        __aicore__ inline void InitComputeBilinearTensor(int32_t actualComputeCount);
        __aicore__ inline void ComputeBilinearCommon(int32_t actualComputeCount, const int64_t curGridPointIndex);
        __aicore__ inline void ComputeGridPointIndex(int64_t gridPointIndex);

        __aicore__ inline void CopyIn(const int64_t offset, const int32_t calCount, const int32_t inputIndex);
        __aicore__ inline void CopyOut(const int32_t offset, const int32_t calCount);
        __aicore__ inline void Compute(const int32_t computeCount, const int64_t curGridPointIndex);

        template <typename T1, typename T2>
        __aicore__ inline T1 CeilDiv(T1 a, T2 b) {
            if (b == 0) {
                return 0;
            }
            return (a + b - 1) / b;
        };

        template <typename T1, typename T2>
        __aicore__ inline T1 CeilAlign(T1 a, T2 b) {
            if (b == 0) {
                return 0;
            }
            return (a + b - 1) / b * b;
        };

    private:
        TPipe* pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> dataInQueue[INPUT_NUM];
        TQue<QuePosition::VECOUT, BUFFER_NUM> dataOutQueue[OUTPUT_NUM];

        TBuf<TPosition::VECCALC> xCoordinateBuf, yCoordinateBuf, zCoordinateBuf;
        TBuf<TPosition::VECCALC> xGradInBuf, yGradInBuf, zGradInBuf;
        TBuf<TPosition::VECCALC> ixtNwBuf, ixtNwIntBuf, tnwBuf;
        TBuf<TPosition::VECCALC> tmp1Buf, tmp2Buf, tmp3Buf;
        TBuf<TPosition::VECCALC> bufferMaskXBuf, bufferMaskYBuf, bufferMaskZBuf, mask1Buf, mask2Buf;
        TBuf<TPosition::VECCALC> computeIndexBuf, computeIndexBuf1, computeIndexBuf2, computeIndexBuf3, computeIndexBuf4,
                                computeIndexBuf5, computeIndexBuf6, computeIndexBuf7, computeIndexBuf8, computeIndexBuf9,
                                computeIndexBuf10, computeIndexBuf11, computeIndexBuf12, computeIndexBuf13, computeIndexBuf14,
                                computeIndexBuf15, computeIndexBuf16, computeIndexBuf17, computeIndexBuf18;
        TBuf<TPosition::VECCALC> gixBuf, giyBuf, gizBuf;
        TBuf<TPosition::VECCALC> sumXBuf, sumYBuf, sumZBuf;
        TBuf<TPosition::VECCALC> dupOneBuf, selBuf1, selBuf2, selBuf3, selBuf4, selBuf5, selBuf6, selBuf7, selBuf8;

        GlobalTensor<T> inputGm[GM_PARAMS_SIZE];

        LocalTensor<uint8_t> mask1Tensor, mask2Tensor;
        LocalTensor<uint16_t> int8ToInt16Mask1, int8ToInt16Mask2;
        LocalTensor<int32_t> tmpIndex1, tmpIndex2;
        LocalTensor<T> dupOneTensor, selTensor1, selTensor2, selTensor3, selTensor4, selTensor5, selTensor6, selTensor7, selTensor8;
        LocalTensor<T> tmp1Tensor, tmp2Tensor, tmp3Tensor;
        LocalTensor<T> gixLocalTensor, giyLocalTensor, gizLocalTensor;
        LocalTensor<T> sumX, sumY, sumZ;
        LocalTensor<T> xTensor, yTensor, zTensor;
        LocalTensor<T> xGradIn, yGradIn, zGradIn;
        LocalTensor<T> inputCoordinate, dstLocal;
        LocalTensor<uint32_t> xPattern, yPattern, zPattern;

        LocalTensor<T> ixtNw, iytNw, iztNw, ixtNe, iytNe, iztNe, ixtSw, iytSw, iztSw,
                    ixtSe, iytSe, iztSe, ixbNw, iybNw, izbNw, ixbNe, iybNe, izbNe,
                    ixbSw, iybSw, izbSw, ixbSe, iybSe, izbSe;
        LocalTensor<int32_t> ixtNwInt, iytNwInt, iztNwInt, ixtNeInt, iytNeInt, iztNeInt,
                            ixtSwInt, iytSwInt, iztSwInt, ixtSeInt, iytSeInt, iztSeInt,
                            ixbNwInt, iybNwInt, izbNwInt, ixbNeInt, iybNeInt, izbNeInt,
                            ixbSwInt, iybSwInt, izbSwInt, ixbSeInt, iybSeInt, izbSeInt;
        LocalTensor<T> tnw, tne, tsw, tse, bnw, bne, bsw, bse;
        LocalTensor<int32_t> tNwIndex, tNeIndex, tSwIndex, tSeIndex, bNwIndex, bNeIndex,
                            bSwIndex, bSeIndex, tnwIndex2, tneIndex2, tswIndex2, tseIndex2,
                            bnwIndex2, bneIndex2, bswIndex2, bseIndex2;

        const int64_t BLOCK_SIZE = 32;

        uint32_t batch = 0;
        int32_t channel = 0;
        int32_t depth = 0;
        int32_t height = 0;
        int32_t width = 0;
        uint32_t gridD = 0;
        uint32_t gridH = 0;
        uint32_t gridW = 0;
        uint32_t interpolation = 0;
        uint32_t padding = 0;
        bool alignCorners = false;
        uint32_t blockNum = 0;
        uint32_t pNumPerCore = 0;
        uint32_t tailPNum = 0;
        uint32_t perBlockCount = 0;
        uint32_t outD = 0;
        uint32_t outH = 0;
        uint32_t outW = 0;
        uint32_t calcCountPerLoop = 0;
        uint32_t maskSize = 0;
        uint32_t maskNum = 0;
        uint32_t blockIdx = 0;
        uint32_t alignChannel = 0;
        uint32_t group = 0;

        int32_t inputStrideN = 0;
        int32_t inputStrideD = 0;
        int32_t inputStrideH = 0;
        int32_t inputStrideW = 0;
        int32_t gradStrideN = 0;
        int32_t gradStrideC = 0;
        int32_t gradStrideD = 0;
        int32_t gradStrideH = 0;
        int32_t gradStrideW = 0;
        int32_t xStrideC = 0;
        int32_t dxStrideN = 0;
        int32_t dxStrideC = 0;
        int32_t dxStrideD = 0;
        int32_t dxStrideH = 0;
        int32_t dxStrideW = 0;
        int64_t pointIndex = 0;
        int64_t gradGmOffset = 0;
        int64_t xGmOffset = 0;
        int32_t ncOffset = 0;

        float fwidth = 0;
        float fheight = 0;
        float fdepth = 0;

        int64_t n = 0;
        int64_t d = 0;
        int64_t h = 0;
        int64_t w = 0;

        int64_t gridPointIndex = 0;
        int64_t ncBaseOffset = 0;

        T gix = static_cast<T>(0);
        T giy = static_cast<T>(0);
        T giz = static_cast<T>(0);
        };

    template <typename T>
    __aicore__ inline void KernelGridSampler3dGrad<T>::Init(GridSampler3dGradV1TilingData* tilingData,
                                                        GM_ADDR inputTensors[GM_PARAMS_SIZE + 1], TPipe* pipe) {
        batch = tilingData->batch;
        channel = tilingData->channel;
        depth = tilingData->depth;
        height = tilingData->height;
        width = tilingData->width;
        gridD = tilingData->gridD;
        gridH = tilingData->gridH;
        gridW = tilingData->gridW;
        interpolation = tilingData->interpolation;
        padding = tilingData->padding;
        alignCorners = tilingData->alignCorners;
        blockNum = tilingData->blockNum;
        pNumPerCore = tilingData->pNumPerCore;
        tailPNum = tilingData->tailPNum;
        calcCountPerLoop = tilingData->ubFactorElement;
        group = tilingData->group;
        blockIdx = GetBlockIdx();

        outD = gridD;
        outH = gridH;
        outW = gridW;
        inputStrideN = width * height * depth * channel;
        inputStrideD = width * height;
        inputStrideH = width;
        inputStrideW = 1;
        gradStrideN = outD * outH * outW * channel;
        gradStrideC = outD * outH * outW;
        gradStrideD = outH * outW;
        gradStrideH = outW;
        gradStrideW = 1;
        xStrideC = depth * height * width;
        dxStrideN = depth * height * width * channel;
        dxStrideC = depth * height * width;
        dxStrideD = height * width;
        dxStrideH = width;
        dxStrideW = 1;

        fdepth = static_cast<T>(depth);
        fheight = static_cast<T>(height);
        fwidth = static_cast<T>(width);
        maskSize = CeilAlign(CeilDiv(calcCountPerLoop, UINT8_BITS), BLOCK_BYTES);
        maskNum = maskSize / sizeof(uint8_t);
        perBlockCount = BLOCK_BYTES / sizeof(T);
        alignChannel = CeilAlign(channel, perBlockCount);

        inputGm[GRAD_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[GRAD_INPUT_INDEX]));
        inputGm[X_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[X_INPUT_INDEX]));
        inputGm[GRID_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[GRID_INPUT_INDEX]));
        inputGm[DX_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[DX_INPUT_INDEX]));
        inputGm[DGRID_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[DGRID_INPUT_INDEX]));

        InitBilinearBuffer(pipe);
        InitBilinearLocalTensor();
    }

    template <typename T>
    __aicore__ inline void KernelGridSampler3dGrad<T>::InitBilinearBuffer(TPipe* pipe) {
        pipe->InitBuffer(dataInQueue[GRAD_INPUT_INDEX], BUFFER_NUM, alignChannel * sizeof(T));
        pipe->InitBuffer(dataInQueue[X_INPUT_INDEX], BUFFER_NUM, alignChannel * sizeof(T));
        pipe->InitBuffer(dataInQueue[GRID_INPUT_INDEX], BUFFER_NUM, 3 * calcCountPerLoop * sizeof(T));
        pipe->InitBuffer(dataOutQueue[DX_OUTPUT_INDEX], BUFFER_NUM, alignChannel * sizeof(T));
        pipe->InitBuffer(dataOutQueue[DGRID_OUTPUT_INDEX], BUFFER_NUM, 3 * calcCountPerLoop * sizeof(T));

        pipe->InitBuffer(xCoordinateBuf, (calcCountPerLoop + ELE_NUM_PER_REPEAT) * sizeof(T));
        pipe->InitBuffer(yCoordinateBuf, (calcCountPerLoop + ELE_NUM_PER_REPEAT) * sizeof(T));
        pipe->InitBuffer(zCoordinateBuf, (calcCountPerLoop + ELE_NUM_PER_REPEAT) * sizeof(T));

        pipe->InitBuffer(xGradInBuf, calcCountPerLoop * sizeof(T));
        pipe->InitBuffer(yGradInBuf, calcCountPerLoop * sizeof(T));
        pipe->InitBuffer(zGradInBuf, calcCountPerLoop * sizeof(T));

        pipe->InitBuffer(bufferMaskXBuf, BLOCK_SIZE * 3);
        pipe->InitBuffer(bufferMaskYBuf, BLOCK_SIZE * 3);
        pipe->InitBuffer(bufferMaskZBuf, BLOCK_SIZE * 3);

        pipe->InitBuffer(mask1Buf, maskSize);
        pipe->InitBuffer(mask2Buf, maskSize);

        pipe->InitBuffer(ixtNwBuf, calcCountPerLoop * sizeof(T) * 24);

        pipe->InitBuffer(ixtNwIntBuf, calcCountPerLoop * sizeof(T) * 24);

        pipe->InitBuffer(tnwBuf, calcCountPerLoop * sizeof(T) * 8);

        pipe->InitBuffer(tmp1Buf, calcCountPerLoop * sizeof(T));
        pipe->InitBuffer(tmp2Buf, calcCountPerLoop * sizeof(T));
        pipe->InitBuffer(tmp3Buf, calcCountPerLoop * sizeof(T));

        pipe->InitBuffer(dupOneBuf, calcCountPerLoop * sizeof(T));
        pipe->InitBuffer(selBuf1, calcCountPerLoop * sizeof(T));
        pipe->InitBuffer(selBuf2, calcCountPerLoop * sizeof(T));
        pipe->InitBuffer(selBuf3, calcCountPerLoop * sizeof(T));
        pipe->InitBuffer(selBuf4, calcCountPerLoop * sizeof(T));
        pipe->InitBuffer(selBuf5, calcCountPerLoop * sizeof(T));
        pipe->InitBuffer(selBuf6, calcCountPerLoop * sizeof(T));
        pipe->InitBuffer(selBuf7, calcCountPerLoop * sizeof(T));
        pipe->InitBuffer(selBuf8, calcCountPerLoop * sizeof(T));

        pipe->InitBuffer(computeIndexBuf1, calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf2, calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf3, calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf4, calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf5, calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf6, calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf7, calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf8, calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf9, calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf10, calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf11, calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf12, calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf13, calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf14, calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf15, calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf16, calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf17, calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf18, calcCountPerLoop * sizeof(int32_t));

        pipe->InitBuffer(gixBuf, alignChannel * sizeof(T));
        pipe->InitBuffer(giyBuf, alignChannel * sizeof(T));
        pipe->InitBuffer(gizBuf, alignChannel * sizeof(T));
        pipe->InitBuffer(sumXBuf, alignChannel * sizeof(T));
        pipe->InitBuffer(sumYBuf, alignChannel * sizeof(T));
        pipe->InitBuffer(sumZBuf, alignChannel * sizeof(T));
    }

    template <typename T>
    __aicore__ inline void KernelGridSampler3dGrad<T>::InitBilinearLocalTensor() {
        mask1Tensor = mask1Buf.Get<uint8_t>(maskNum);
        mask2Tensor = mask2Buf.Get<uint8_t>(maskNum);

        dupOneTensor = dupOneBuf.Get<T>(calcCountPerLoop);
        tmpIndex1 = computeIndexBuf1.Get<int32_t>(calcCountPerLoop);
        tmpIndex2 = computeIndexBuf18.Get<int32_t>(calcCountPerLoop);

        selTensor1 = selBuf1.Get<T>(calcCountPerLoop);
        selTensor2 = selBuf2.Get<T>(calcCountPerLoop);
        selTensor3 = selBuf3.Get<T>(calcCountPerLoop);
        selTensor4 = selBuf4.Get<T>(calcCountPerLoop);
        selTensor5 = selBuf5.Get<T>(calcCountPerLoop);
        selTensor6 = selBuf6.Get<T>(calcCountPerLoop);
        selTensor7 = selBuf7.Get<T>(calcCountPerLoop);
        selTensor8 = selBuf8.Get<T>(calcCountPerLoop);
        tmp1Tensor = tmp1Buf.Get<T>(calcCountPerLoop);
        tmp2Tensor = tmp2Buf.Get<T>(calcCountPerLoop);
        tmp3Tensor = tmp3Buf.Get<T>(calcCountPerLoop);
        sumX = sumXBuf.Get<T>(alignChannel);
        sumY = sumYBuf.Get<T>(alignChannel);
        sumZ = sumZBuf.Get<T>(alignChannel);

        tnw = tnwBuf.Get<T>();
        tne = tnw[calcCountPerLoop * 1];
        tsw = tnw[calcCountPerLoop * 2];
        tse = tnw[calcCountPerLoop * 3];
        bnw = tnw[calcCountPerLoop * 4];
        bne = tnw[calcCountPerLoop * 5];
        bsw = tnw[calcCountPerLoop * 6];
        bse = tnw[calcCountPerLoop * 7];

        tNwIndex = computeIndexBuf2.Get<int32_t>(calcCountPerLoop);
        tNeIndex = computeIndexBuf3.Get<int32_t>(calcCountPerLoop);
        tSwIndex = computeIndexBuf4.Get<int32_t>(calcCountPerLoop);
        tSeIndex = computeIndexBuf5.Get<int32_t>(calcCountPerLoop);
        bNwIndex = computeIndexBuf6.Get<int32_t>(calcCountPerLoop);
        bNeIndex = computeIndexBuf7.Get<int32_t>(calcCountPerLoop);
        bSwIndex = computeIndexBuf8.Get<int32_t>(calcCountPerLoop);
        bSeIndex = computeIndexBuf9.Get<int32_t>(calcCountPerLoop);
        tnwIndex2 = computeIndexBuf10.Get<int32_t>(calcCountPerLoop);
        tneIndex2 = computeIndexBuf11.Get<int32_t>(calcCountPerLoop);
        tswIndex2 = computeIndexBuf12.Get<int32_t>(calcCountPerLoop);
        tseIndex2 = computeIndexBuf13.Get<int32_t>(calcCountPerLoop);
        bnwIndex2 = computeIndexBuf14.Get<int32_t>(calcCountPerLoop);
        bneIndex2 = computeIndexBuf15.Get<int32_t>(calcCountPerLoop);
        bswIndex2 = computeIndexBuf16.Get<int32_t>(calcCountPerLoop);
        bseIndex2 = computeIndexBuf17.Get<int32_t>(calcCountPerLoop);

        gixLocalTensor = gixBuf.Get<T>(alignChannel);
        giyLocalTensor = giyBuf.Get<T>(alignChannel);
        gizLocalTensor = gizBuf.Get<T>(alignChannel);

        ixtNw = ixtNwBuf.Get<T>();
        iytNw = ixtNw[calcCountPerLoop * 1];
        iztNw = ixtNw[calcCountPerLoop * 2];
        ixtNe = ixtNw[calcCountPerLoop * 3];
        iytNe = ixtNw[calcCountPerLoop * 4];
        iztNe = ixtNw[calcCountPerLoop * 5];
        ixtSw = ixtNw[calcCountPerLoop * 6];
        iytSw = ixtNw[calcCountPerLoop * 7];
        iztSw = ixtNw[calcCountPerLoop * 8];
        ixtSe = ixtNw[calcCountPerLoop * 9];
        iytSe = ixtNw[calcCountPerLoop * 10];
        iztSe = ixtNw[calcCountPerLoop * 11];
        ixbNw = ixtNw[calcCountPerLoop * 12];
        iybNw = ixtNw[calcCountPerLoop * 13];
        izbNw = ixtNw[calcCountPerLoop * 14];
        ixbNe = ixtNw[calcCountPerLoop * 15];
        iybNe = ixtNw[calcCountPerLoop * 16];
        izbNe = ixtNw[calcCountPerLoop * 17];
        ixbSw = ixtNw[calcCountPerLoop * 18];
        iybSw = ixtNw[calcCountPerLoop * 19];
        izbSw = ixtNw[calcCountPerLoop * 20];
        ixbSe = ixtNw[calcCountPerLoop * 21];
        iybSe = ixtNw[calcCountPerLoop * 22];
        izbSe = ixtNw[calcCountPerLoop * 23];

        ixtNwInt = ixtNwIntBuf.Get<int32_t>();
        iytNwInt = ixtNwInt[calcCountPerLoop * 1];
        iztNwInt = ixtNwInt[calcCountPerLoop * 2];
        ixtNeInt = ixtNwInt[calcCountPerLoop * 3];
        iytNeInt = ixtNwInt[calcCountPerLoop * 4];
        iztNeInt = ixtNwInt[calcCountPerLoop * 5];
        ixtSwInt = ixtNwInt[calcCountPerLoop * 6];
        iytSwInt = ixtNwInt[calcCountPerLoop * 7];
        iztSwInt = ixtNwInt[calcCountPerLoop * 8];
        ixtSeInt = ixtNwInt[calcCountPerLoop * 9];
        iytSeInt = ixtNwInt[calcCountPerLoop * 10];
        iztSeInt = ixtNwInt[calcCountPerLoop * 11];
        ixbNwInt = ixtNwInt[calcCountPerLoop * 12];
        iybNwInt = ixtNwInt[calcCountPerLoop * 13];
        izbNwInt = ixtNwInt[calcCountPerLoop * 14];
        ixbNeInt = ixtNwInt[calcCountPerLoop * 15];
        iybNeInt = ixtNwInt[calcCountPerLoop * 16];
        izbNeInt = ixtNwInt[calcCountPerLoop * 17];
        ixbSwInt = ixtNwInt[calcCountPerLoop * 18];
        iybSwInt = ixtNwInt[calcCountPerLoop * 19];
        izbSwInt = ixtNwInt[calcCountPerLoop * 20];
        ixbSeInt = ixtNwInt[calcCountPerLoop * 21];
        iybSeInt = ixtNwInt[calcCountPerLoop * 22];
        izbSeInt = ixtNwInt[calcCountPerLoop * 23];
    }

    template <typename T>
    __aicore__ inline void KernelGridSampler3dGrad<T>::DupValue() {
        Duplicate<T>(dupOneTensor, 1, calcCountPerLoop);
        Duplicate<T>(sumX, 0, alignChannel);
        Duplicate<T>(sumY, 0, alignChannel);
        Duplicate<T>(sumZ, 0, alignChannel);
    }

    template <typename T>
    __aicore__ inline void KernelGridSampler3dGrad<T>::ComputeWeight(
        LocalTensor<T> dst, LocalTensor<T> xCoorTensor1, LocalTensor<T> xCoorTensor2, LocalTensor<T> yCoorTensor1,
        LocalTensor<T> yCoorTensor2, LocalTensor<T> zCoorTensor1, LocalTensor<T> zCoorTensor2, const int32_t calCount) {
        Muls(tmp1Tensor, xCoorTensor1, static_cast<T>(-1), calCount);
        Add(tmp1Tensor, xCoorTensor2, tmp1Tensor, calCount);
        Muls(tmp2Tensor, yCoorTensor1, static_cast<T>(-1), calCount);
        Add(tmp2Tensor, yCoorTensor2, tmp2Tensor, calCount);
        Muls(tmp3Tensor, zCoorTensor1, static_cast<T>(-1), calCount);
        Add(tmp3Tensor, zCoorTensor2, tmp3Tensor, calCount);
        Mul(dst, tmp1Tensor, tmp2Tensor, calCount);
        Mul(dst, dst, tmp3Tensor, calCount);
    }

    template <typename T>
    __aicore__ inline void KernelGridSampler3dGrad<T>::ComputeSourceIndexSetGrad(LocalTensor<T> dataTensor,
                                                                            LocalTensor<T> dupTensor, const T size,
                                                                            const int32_t calCount) {
        if (alignCorners) {
            T val = static_cast<T>(size - 1) / 2;
            Duplicate<T>(dupTensor, val, calCount);
            Adds(dataTensor, dataTensor, static_cast<T>(1), calCount);
            Muls(dataTensor, dataTensor, static_cast<T>(0.5), calCount);
            Muls(dataTensor, dataTensor, static_cast<T>(size - 1), calCount);
        } else {
            T val = static_cast<T>(size) / 2;
            Duplicate<T>(dupTensor, val, calCount);
            Adds(dataTensor, dataTensor, static_cast<T>(1), calCount);
            Muls(dataTensor, dataTensor, static_cast<T>(size), calCount);
            Adds(dataTensor, dataTensor, static_cast<T>(-1), calCount);
            Muls(dataTensor, dataTensor, static_cast<T>(0.5), calCount);
        }

        int32_t newCalCount =
            ((calCount * FLOAT_BYTES - 1 + ALIGN_256_BYTES) / ALIGN_256_BYTES * ALIGN_256_BYTES) / FLOAT_BYTES;

        // If the data is inf/-inf/nan, convert the data to -100.
        CompareScalar(mask1Tensor, dataTensor, static_cast<T>(INT_MAX - 1), CMPMODE::LE, newCalCount);
        Select(dataTensor, mask1Tensor, dataTensor, static_cast<T>(-100.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
        CompareScalar(mask1Tensor, dataTensor, static_cast<T>(INT_MIN), CMPMODE::GE, newCalCount);
        Select(dataTensor, mask1Tensor, dataTensor, static_cast<T>(-100.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
        Compare(mask1Tensor, dataTensor, dataTensor, CMPMODE::EQ, newCalCount);
        Select(dataTensor, mask1Tensor, dataTensor, static_cast<T>(-100.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
    }

    template <typename T>
    __aicore__ inline void KernelGridSampler3dGrad<T>::WithinBounds3d(
    LocalTensor<T> dst, LocalTensor<T> izT, LocalTensor<T> iyT, LocalTensor<T> ixT, LocalTensor<T> weight, const int32_t calCount) {
        int32_t newCalCount = ((calCount * FLOAT_BYTES + ALIGN_256_BYTES - 1) / ALIGN_256_BYTES * ALIGN_256_BYTES) /
                                FLOAT_BYTES;

        CompareScalar(mask1Tensor, izT, static_cast<T>(0), CMPMODE::GE, newCalCount);
        CompareScalar(mask2Tensor, izT, static_cast<T>(fdepth), CMPMODE::LT, newCalCount);
        int8ToInt16Mask1 = mask1Tensor.ReinterpretCast<uint16_t>();
        int8ToInt16Mask2 = mask2Tensor.ReinterpretCast<uint16_t>();

        And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);
        CompareScalar(mask1Tensor, iyT, static_cast<T>(0), CMPMODE::GE, newCalCount);
        And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);
        CompareScalar(mask1Tensor, iyT, static_cast<T>(fheight), CMPMODE::LT, newCalCount);
        And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);

        CompareScalar(mask1Tensor, ixT, static_cast<T>(0), CMPMODE::GE, newCalCount);
        And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);
        CompareScalar(mask1Tensor, ixT, static_cast<T>(fwidth), CMPMODE::LT, newCalCount);
        And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);

        Select(dst, int8ToInt16Mask2, dupOneTensor, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
        Select(weight, int8ToInt16Mask2, weight, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
    }

    template <typename T>
    __aicore__ inline void KernelGridSampler3dGrad<T>::ComputeIndex(LocalTensor<int32_t> dstIndex,
                                                                                    LocalTensor<int32_t> dstIndex2,
                                                                                    LocalTensor<int32_t> zCoor,
                                                                                    LocalTensor<int32_t> yCoor,
                                                                                    LocalTensor<int32_t> xCoor,
                                                                                    const int32_t calCount) {
        Mins(zCoor, zCoor, depth - 1, calCount);
        Maxs(zCoor, zCoor, 0, calCount);
        Mins(yCoor, yCoor, height - 1, calCount);
        Maxs(yCoor, yCoor, 0, calCount);
        Mins(xCoor, xCoor, width - 1, calCount);
        Maxs(xCoor, xCoor, 0, calCount);

        Muls(tmpIndex1, zCoor, inputStrideD, calCount);
        Muls(dstIndex, yCoor, inputStrideH, calCount);
        Add(tmpIndex1, tmpIndex1, dstIndex, calCount);
        Add(dstIndex, tmpIndex1, xCoor, calCount);
        Muls(dstIndex, dstIndex, channel, calCount);

        Muls(tmpIndex1, zCoor, dxStrideD, calCount);
        Muls(dstIndex2, yCoor, dxStrideH, calCount);
        Add(tmpIndex1, tmpIndex1, dstIndex2, calCount);
        Add(dstIndex2, tmpIndex1, xCoor, calCount);
        Muls(dstIndex2, dstIndex2, channel, calCount);
    }

    template <typename T>
    __aicore__ inline void KernelGridSampler3dGrad<T>::ComputeAfterTransposeGridGrad(
    LocalTensor<int32_t> srcIndex, LocalTensor<T> zCoor1, LocalTensor<T> zCoor2, LocalTensor<T> yCoor1,
    LocalTensor<T> yCoor2, LocalTensor<T> xCoor1, LocalTensor<T> xCoor2, LocalTensor<T> gOutLocalTensor,
    LocalTensor<T> selTensor, const int32_t coorIndex, const int32_t batchIdx, int32_t xTag, int32_t yTag,
    int32_t zTag) {
        pointIndex = srcIndex.GetValue(coorIndex);
        xGmOffset = batchIdx * inputStrideN + pointIndex;
        T xVal = xCoor1.GetValue(coorIndex) - xCoor2.GetValue(coorIndex);
        T yVal = yCoor1.GetValue(coorIndex) - yCoor2.GetValue(coorIndex);
        T zVal = zCoor1.GetValue(coorIndex) - zCoor2.GetValue(coorIndex);
        T coorValue = selTensor.GetValue(coorIndex);

        LocalTensor<T> inputXLocalTensor = dataInQueue[X_INPUT_INDEX].AllocTensor<T>();
        DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
        DataCopyPadExtParams padParams = {true, 0, 0, static_cast<T>(0)};
        copyParams.blockLen = channel * sizeof(T);
        padParams.rightPadding = alignChannel - channel;

        DataCopyPad(inputXLocalTensor, inputGm[X_INPUT_INDEX][xGmOffset], copyParams, padParams);

        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);

        Muls(gixLocalTensor, inputXLocalTensor, yVal, channel);
        Muls(gixLocalTensor, gixLocalTensor, zVal, channel);
        Mul(gixLocalTensor, gOutLocalTensor, gixLocalTensor, channel);
        Muls(gixLocalTensor, gixLocalTensor, coorValue, channel);

        Muls(giyLocalTensor, inputXLocalTensor, xVal, channel);
        Muls(giyLocalTensor, giyLocalTensor, zVal, channel);
        Mul(giyLocalTensor, gOutLocalTensor, giyLocalTensor, channel);
        Muls(giyLocalTensor, giyLocalTensor, coorValue, channel);

        Muls(gizLocalTensor, inputXLocalTensor, xVal, channel);
        Muls(gizLocalTensor, gizLocalTensor, yVal, channel);
        Mul(gizLocalTensor, gOutLocalTensor, gizLocalTensor, channel);
        Muls(gizLocalTensor, gizLocalTensor, coorValue, channel);

        Axpy(sumZ, gizLocalTensor, static_cast<T>(zTag), channel);
        Axpy(sumY, giyLocalTensor, static_cast<T>(yTag), channel);
        Axpy(sumX, gixLocalTensor, static_cast<T>(xTag), channel);

        dataInQueue[X_INPUT_INDEX].FreeTensor(inputXLocalTensor);
    }

    template <typename T>
    __aicore__ inline void KernelGridSampler3dGrad<T>::ComputeAfterTransposeXGrad(LocalTensor<int32_t> srcIndex,
                                                                            LocalTensor<T> weight,
                                                                            const int32_t coorIndex,
                                                                            const int64_t ncOffset,
                                                                            LocalTensor<T> gOutLocalTensor) {
        T weightVal = weight.GetValue(coorIndex);
        int64_t offset = ncOffset + srcIndex.GetValue(coorIndex);
        LocalTensor<T> localTensor = dataOutQueue[DX_OUTPUT_INDEX].AllocTensor<T>();
        event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIDSToV);
        WaitFlag<HardEvent::S_V>(eventIDSToV);

        Muls(localTensor, gOutLocalTensor, weightVal, alignChannel);
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);

        DataCopyExtParams copyParams{1, 0, 0, 0, 0};
        copyParams.blockLen = channel * sizeof(T);
        SetAtomicAdd<T>();
        DataCopyPad(inputGm[DX_INPUT_INDEX][offset], localTensor, copyParams);
        SetAtomicNone();
        dataOutQueue[DX_OUTPUT_INDEX].FreeTensor(localTensor);
    }


    template <typename T>
    __aicore__ inline void KernelGridSampler3dGrad<T>::CopyIn(const int64_t offset, const int32_t calCount,
                                                        const int32_t inputIndex) {
        LocalTensor<T> dataLocal = dataInQueue[inputIndex].AllocTensor<T>();
        DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
        DataCopyPadExtParams padParams = {true, 0, 0, static_cast<T>(0)};

        int32_t alignCalCount = CeilAlign(calCount, perBlockCount);
        copyParams.blockLen = calCount * sizeof(T);
        padParams.rightPadding = alignCalCount - calCount;

        DataCopyPad(dataLocal, inputGm[inputIndex][offset], copyParams, padParams);
        dataInQueue[inputIndex].EnQue(dataLocal);
    }

    template <typename T>
    __aicore__ inline void KernelGridSampler3dGrad<T>::CopyOut(const int32_t offset, const int32_t calCount) {
        LocalTensor<T> dstLocal = dataOutQueue[DGRID_OUTPUT_INDEX].DeQue<T>();
        DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
        copyParams.blockLen = calCount * sizeof(T);
        DataCopyPad(inputGm[DGRID_INPUT_INDEX][offset], dstLocal, copyParams);
        dataOutQueue[DGRID_OUTPUT_INDEX].FreeTensor(dstLocal);
    }

    template <typename T>
    __aicore__ inline void KernelGridSampler3dGrad<T>::Compute(const int32_t computeCount, const int64_t curGridPointIndex) {
        int32_t actualComputeCount = computeCount / 3;
        uint32_t mask = ELE_NUM_PER_REPEAT;
        uint64_t rsvdCnt = 0;
        bool reduceMode = true;
        uint8_t src0BlockStride = 1;
        uint16_t repeatTimes = CeilDiv(computeCount, ELE_NUM_PER_REPEAT);
        uint8_t src0RepeatStride = REPEAT_STRIDE_0;
        uint8_t src1RepeatStride = REPEAT_STRIDE_1;

        InitComputeTensor();

        DupValue();

        GatherMask(xTensor, inputCoordinate, xPattern, reduceMode, GATHER_MASK_NUM,
                    {1, repeatTimes, src0RepeatStride, src1RepeatStride}, rsvdCnt);
        GatherMask(yTensor, inputCoordinate, yPattern, reduceMode, GATHER_MASK_NUM,
                    {1, repeatTimes, src0RepeatStride, src1RepeatStride}, rsvdCnt);
        GatherMask(zTensor, inputCoordinate, zPattern, reduceMode, GATHER_MASK_NUM,
                    {1, repeatTimes, src0RepeatStride, src1RepeatStride}, rsvdCnt);

        ComputeSourceIndexSetGrad(xTensor, xGradIn, fwidth, actualComputeCount);
        ComputeSourceIndexSetGrad(yTensor, yGradIn, fheight, actualComputeCount);
        ComputeSourceIndexSetGrad(zTensor, zGradIn, fdepth, actualComputeCount);

        ComputeBilinear(actualComputeCount, curGridPointIndex);

        dataOutQueue[DGRID_OUTPUT_INDEX].EnQue(dstLocal);
        dataInQueue[GRID_INPUT_INDEX].FreeTensor(inputCoordinate);
    }

    template <typename T>
    __aicore__ inline void KernelGridSampler3dGrad<T>::InitComputeTensor() {
        xTensor = xCoordinateBuf.Get<T>(calcCountPerLoop + ELE_NUM_PER_REPEAT);
        yTensor = yCoordinateBuf.Get<T>(calcCountPerLoop + ELE_NUM_PER_REPEAT);
        zTensor = zCoordinateBuf.Get<T>(calcCountPerLoop + ELE_NUM_PER_REPEAT);

        xGradIn = xGradInBuf.Get<T>(calcCountPerLoop);
        yGradIn = yGradInBuf.Get<T>(calcCountPerLoop);
        zGradIn = zGradInBuf.Get<T>(calcCountPerLoop);

        inputCoordinate = dataInQueue[GRID_INPUT_INDEX].DeQue<T>();
        dstLocal = dataOutQueue[DGRID_OUTPUT_INDEX].AllocTensor<T>();

        xPattern = bufferMaskXBuf.Get<uint32_t>(3);
        yPattern = bufferMaskYBuf.Get<uint32_t>(3);
        zPattern = bufferMaskZBuf.Get<uint32_t>(3);

        // Set pattern values for x to select first element of three
        xPattern.SetValue(0, 0b1001001001001001001001001001001);
        xPattern.SetValue(1, 0b10010010010010010010010010010010);
        xPattern.SetValue(2, 0b100100100100100100100100100100);

        // Set pattern values for y to select second element of three
        yPattern.SetValue(0, 0b10010010010010010010010010010010);
        yPattern.SetValue(1, 0b100100100100100100100100100100);
        yPattern.SetValue(2, 0b1001001001001001001001001001001);

        // Set pattern values for z to select third element of three
        zPattern.SetValue(0, 0b100100100100100100100100100100);
        zPattern.SetValue(1, 0b1001001001001001001001001001001);
        zPattern.SetValue(2, 0b10010010010010010010010010010010);
    }

    template <typename T>
    __aicore__ inline void KernelGridSampler3dGrad<T>::ComputeBilinear(int32_t actualComputeCount,
                                                                const int64_t curGridPointIndex) {
        InitComputeBilinearTensor(actualComputeCount);

        ComputeWeight(tnw, xTensor, ixbSe, yTensor, iybSe, zTensor, izbSe, actualComputeCount);
        ComputeWeight(tne, ixbSw, xTensor, yTensor, iybSw, zTensor, izbSw, actualComputeCount);
        ComputeWeight(tsw, xTensor, ixbNe, iybNe, yTensor, zTensor, izbNe, actualComputeCount);
        ComputeWeight(tse, ixbNw, xTensor, iybNw, yTensor, zTensor, izbNw, actualComputeCount);
        ComputeWeight(bnw, xTensor, ixtSe, yTensor, iytSe, iztSe, zTensor, actualComputeCount);
        ComputeWeight(bne, ixtSw, xTensor, yTensor, iytSw, iztSw, zTensor, actualComputeCount);
        ComputeWeight(bsw, xTensor, ixtNe, iytNe, yTensor, iztNe, zTensor, actualComputeCount);
        ComputeWeight(bse, ixtNw, xTensor, iytNw, yTensor, iztNw, zTensor, actualComputeCount);

        WithinBounds3d(selTensor1, iztNw, iytNw, ixtNw, tnw, actualComputeCount);
        WithinBounds3d(selTensor2, iztNe, iytNe, ixtNe, tne, actualComputeCount);
        WithinBounds3d(selTensor3, iztSw, iytSw, ixtSw, tsw, actualComputeCount);
        WithinBounds3d(selTensor4, iztSe, iytSe, ixtSe, tse, actualComputeCount);
        WithinBounds3d(selTensor5, izbNw, iybNw, ixbNw, bnw, actualComputeCount);
        WithinBounds3d(selTensor6, izbNe, iybNe, ixbNe, bne, actualComputeCount);
        WithinBounds3d(selTensor7, izbSw, iybSw, ixbSw, bsw, actualComputeCount);
        WithinBounds3d(selTensor8, izbSe, iybSe, ixbSe, bse, actualComputeCount);

        ComputeIndex(tNwIndex, tnwIndex2, iztNwInt, iytNwInt, ixtNwInt, actualComputeCount);
        ComputeIndex(tNeIndex, tneIndex2, iztNeInt, iytNeInt, ixtNeInt, actualComputeCount);
        ComputeIndex(tSwIndex, tswIndex2, iztSwInt, iytSwInt, ixtSwInt, actualComputeCount);
        ComputeIndex(tSeIndex, tseIndex2, iztSeInt, iytSeInt, ixtSeInt, actualComputeCount);
        ComputeIndex(bNwIndex, bnwIndex2, izbNwInt, iybNwInt, ixbNwInt, actualComputeCount);
        ComputeIndex(bNeIndex, bneIndex2, izbNeInt, iybNeInt, ixbNeInt, actualComputeCount);
        ComputeIndex(bSwIndex, bswIndex2, izbSwInt, iybSwInt, ixbSwInt, actualComputeCount);
        ComputeIndex(bSeIndex, bseIndex2, izbSeInt, iybSeInt, ixbSeInt, actualComputeCount);

        ComputeBilinearCommon(actualComputeCount, curGridPointIndex);
    }

    template <typename T>
    __aicore__ inline void KernelGridSampler3dGrad<T>::InitComputeBilinearTensor(int32_t actualComputeCount) {

        Cast(ixtNwInt, xTensor, RoundMode::CAST_FLOOR, actualComputeCount);
        Cast(iytNwInt, yTensor, RoundMode::CAST_FLOOR, actualComputeCount);
        Cast(iztNwInt, zTensor, RoundMode::CAST_FLOOR, actualComputeCount);

        Adds(ixtNeInt, ixtNwInt, static_cast<int32_t>(1), actualComputeCount);
        iytNeInt = iytNwInt;
        iztNeInt = iztNwInt;

        ixtSwInt = ixtNwInt;
        Adds(iytSwInt, iytNwInt, static_cast<int32_t>(1), actualComputeCount);
        iztSwInt = iztNwInt;

        Adds(ixtSeInt, ixtNwInt, static_cast<int32_t>(1), actualComputeCount);
        Adds(iytSeInt, iytNwInt, static_cast<int32_t>(1), actualComputeCount);
        iztSeInt = iztNwInt;

        ixbNwInt = ixtNwInt;
        iybNwInt = iytNwInt;
        Adds(izbNwInt, iztNwInt, static_cast<int32_t>(1), actualComputeCount);

        Adds(ixbNeInt, ixtNwInt, static_cast<int32_t>(1), actualComputeCount);
        iybNeInt = iytNwInt;
        Adds(izbNeInt, iztNwInt, static_cast<int32_t>(1), actualComputeCount);

        ixbSwInt = ixtNwInt;
        Adds(iybSwInt, iytNwInt, static_cast<int32_t>(1), actualComputeCount);
        Adds(izbSwInt, iztNwInt, static_cast<int32_t>(1), actualComputeCount);

        Adds(ixbSeInt, ixtNwInt, static_cast<int32_t>(1), actualComputeCount);
        Adds(iybSeInt, iytNwInt, static_cast<int32_t>(1), actualComputeCount);
        Adds(izbSeInt, iztNwInt, static_cast<int32_t>(1), actualComputeCount);

        Cast(ixtNw, ixtNwInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(iytNw, iytNwInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(iztNw, iztNwInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(ixtNe, ixtNeInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(iytNe, iytNeInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(iztNe, iztNeInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(ixtSw, ixtSwInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(iytSw, iytSwInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(iztSw, iztSwInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(ixtSe, ixtSeInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(iytSe, iytSeInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(iztSe, iztSeInt, RoundMode::CAST_NONE, actualComputeCount);

        Cast(ixbNw, ixbNwInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(iybNw, iybNwInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(izbNw, izbNwInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(ixbNe, ixbNeInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(iybNe, iybNeInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(izbNe, izbNeInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(ixbSw, ixbSwInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(iybSw, iybSwInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(izbSw, izbSwInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(ixbSe, ixbSeInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(iybSe, iybSeInt, RoundMode::CAST_NONE, actualComputeCount);
        Cast(izbSe, izbSeInt, RoundMode::CAST_NONE, actualComputeCount);
    }

    template <typename T>
    __aicore__ inline void KernelGridSampler3dGrad<T>::ComputeBilinearCommon(int32_t actualComputeCount,
                                                                        const int64_t curGridPointIndex) {
        for (int32_t i = 0; i < actualComputeCount; i++) {
            gridPointIndex = curGridPointIndex + i;
            ComputeGridPointIndex(gridPointIndex);
            LocalTensor<T> gOutLocalTensor = dataInQueue[GRAD_INPUT_INDEX].AllocTensor<T>();

            DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
            DataCopyPadExtParams padParams = {true, 0, 0, static_cast<T>(0)};

            int32_t alignCalCount = CeilAlign(channel, perBlockCount);
            copyParams.blockLen = channel * sizeof(T);
            padParams.rightPadding = alignCalCount - channel;
            DataCopyPad(gOutLocalTensor, inputGm[GRAD_INPUT_INDEX][gradGmOffset], copyParams, padParams);

            ComputeAfterTransposeGridGrad(tNwIndex, izbSe, zTensor, iybSe, yTensor, ixbSe, xTensor, gOutLocalTensor, selTensor1, i, n, -1, -1, -1);
            ComputeAfterTransposeXGrad(tnwIndex2, tnw, i, ncBaseOffset, gOutLocalTensor);
            ComputeAfterTransposeGridGrad(tNeIndex, izbSw, zTensor, iybSw, yTensor, xTensor, ixbSw, gOutLocalTensor, selTensor2, i, n, 1, -1, -1);
            ComputeAfterTransposeXGrad(tneIndex2, tne, i, ncBaseOffset, gOutLocalTensor);
            ComputeAfterTransposeGridGrad(tSwIndex, izbNe, zTensor, yTensor, iybNe, ixbNe, xTensor, gOutLocalTensor, selTensor3, i, n, -1, 1, -1);
            ComputeAfterTransposeXGrad(tswIndex2, tsw, i, ncBaseOffset, gOutLocalTensor);
            ComputeAfterTransposeGridGrad(tSeIndex, izbNw, zTensor, yTensor, iybNw, xTensor, ixbNw, gOutLocalTensor, selTensor4, i, n, 1, 1, -1);
            ComputeAfterTransposeXGrad(tseIndex2, tse, i, ncBaseOffset, gOutLocalTensor);
            ComputeAfterTransposeGridGrad(bNwIndex, zTensor, iztSe, iytSe, yTensor, ixtSe, xTensor, gOutLocalTensor, selTensor5, i, n, -1, -1, 1);
            ComputeAfterTransposeXGrad(bnwIndex2, bnw, i, ncBaseOffset, gOutLocalTensor);
            ComputeAfterTransposeGridGrad(bNeIndex, zTensor, iztSw, iytSw, yTensor, xTensor, ixtSw, gOutLocalTensor, selTensor6, i, n, 1, -1, 1);
            ComputeAfterTransposeXGrad(bneIndex2, bne, i, ncBaseOffset, gOutLocalTensor);
            ComputeAfterTransposeGridGrad(bSwIndex, zTensor, iztNe, yTensor, iytNe, ixtNe, xTensor, gOutLocalTensor, selTensor7, i, n, -1, 1, 1);
            ComputeAfterTransposeXGrad(bswIndex2, bsw, i, ncBaseOffset, gOutLocalTensor);
            ComputeAfterTransposeGridGrad(bSeIndex, zTensor, iztNw, yTensor, iytNw, xTensor, ixtNw, gOutLocalTensor, selTensor8, i, n, 1, 1, 1);
            ComputeAfterTransposeXGrad(bseIndex2, bse, i, ncBaseOffset, gOutLocalTensor);

            ReduceSum<T>(sumZ, sumZ, sumZ, alignChannel);
            ReduceSum<T>(sumY, sumY, sumY, alignChannel);
            ReduceSum<T>(sumX, sumX, sumX, alignChannel);

            gix += sumX.GetValue(0);
            giy += sumY.GetValue(0);
            giz += sumZ.GetValue(0);

            dstLocal.SetValue(3 * i, gix * xGradIn.GetValue(i));
            dstLocal.SetValue(3 * i + 1, giy * yGradIn.GetValue(i));
            dstLocal.SetValue(3 * i + 2, giz * zGradIn.GetValue(i));

            Duplicate<T>(sumX, 0, alignChannel);
            Duplicate<T>(sumY, 0, alignChannel);
            Duplicate<T>(sumZ, 0, alignChannel);

            gix = static_cast<T>(0);
            giy = static_cast<T>(0);
            giz = static_cast<T>(0);

            dataInQueue[GRAD_INPUT_INDEX].FreeTensor(gOutLocalTensor);
        }
    }


    template <typename T>
    __aicore__ inline void KernelGridSampler3dGrad<T>::ComputeGridPointIndex(int64_t gridPointIndex) {
        w = gridPointIndex % outW;
        h = (gridPointIndex / outW) % outH;
        d = (gridPointIndex / (outH * outW)) % outD;
        n = gridPointIndex / (outD * outH * outW);
        ncBaseOffset = n * dxStrideN;
        gradGmOffset = n * gradStrideN + (d * gradStrideD + h * gradStrideH + w * gradStrideW) * channel;
    }

    template <typename T>
    __aicore__ inline void KernelGridSampler3dGrad<T>::Process() {
        int64_t computePNum = 0;
        int64_t gridGmOffset = 0;
        int64_t gridOffset = 0;
        int64_t cycleOffset = 0;
        int64_t curGridPointIndex = 0;

        if (blockIdx < tailPNum) {
            computePNum = pNumPerCore + 1;
            gridOffset = blockIdx * computePNum;
        } else {
            computePNum = pNumPerCore;
            gridOffset = blockIdx * pNumPerCore + tailPNum;
        }

        int32_t copyCountPerTime = 3 * calcCountPerLoop;
        int32_t copyTimes = CeilDiv(computePNum * 3, copyCountPerTime);
        int32_t actualComputeNum = copyCountPerTime;

        for (int32_t i = 0; i < copyTimes; i++) {
            if (i == copyTimes - 1) {
                actualComputeNum = computePNum * 3 - (copyTimes - 1) * copyCountPerTime;
            }
            cycleOffset = i * copyCountPerTime;
            gridGmOffset = cycleOffset + gridOffset * 3;
            curGridPointIndex = gridOffset + i * copyCountPerTime / 3;

            CopyIn(gridGmOffset, actualComputeNum, GRID_INPUT_INDEX);

            Compute(actualComputeNum, curGridPointIndex);

            CopyOut(gridGmOffset, actualComputeNum);
        }
    }
}  // namespace GridSampler3dGrad

extern "C" __global__ __aicore__ void grid_sampler3d_grad_v1(GM_ADDR grad, GM_ADDR x, GM_ADDR grid, GM_ADDR dx,
                                                        GM_ADDR dgrid, GM_ADDR workspace, GM_ADDR tiling) {
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr) {
        return;
    }
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR gmTensor[6] = {grad, x, grid, dx, dgrid, workspace};
    KernelGridSampler3dGrad<float> op;
    op.Init(&tilingData, gmTensor, &pipe);
    op.Process();
}