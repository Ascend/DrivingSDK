#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;  // tensor num for each queue

class KernelBatchMatmulVector {
public:
    __aicore__ inline KernelBatchMatmulVector() {}

    TBuf<TPosition::VECCALC> projBuf, ptsBuf, pointsBuf, indicesOffsetBuf, indicesPairBuf, tempGmBuf;
    GlobalTensor<DTYPE_PROJECTION_MAT> projectionMatGm;
    GlobalTensor<DTYPE_PTS_EXTEND> ptsExtendGm;
    GlobalTensor<DTYPE_POINT_2D> point2dGm;
    uint64_t coreUsed;
    uint64_t coreData;
    uint64_t copyLoop;
    uint64_t copyTail;
    uint64_t lastCopyLoop;
    uint64_t lastCopyTail;
    uint64_t availableUbSize;
    uint64_t totalResult;
    uint64_t ptsTotal;
    uint64_t dimSizeSecondLast;
    uint64_t dimSizeLast;
    LocalTensor<DTYPE_PROJECTION_MAT> projMatUb;
    LocalTensor<DTYPE_PROJECTION_MAT> ptsUb;
    LocalTensor<DTYPE_PROJECTION_MAT> point2dUb;
    DataCopyPadParams padParams{false, 0, 0, 0};
    int32_t totalKernelSize;
    int32_t dataEachBlock = 8;

    __aicore__ inline void Init(GM_ADDR projectionMat,
                                GM_ADDR ptsExtend,
                                GM_ADDR point2d,
                                GM_ADDR workspace,
                                BatchMatmulVectorTilingData* tilingData, TPipe* pipe)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->coreUsed = tilingData->coreUsed;
        this->coreData = tilingData->coreData;
        this->copyLoop = tilingData->copyLoop;
        this->copyTail = tilingData->copyTail;
        this->lastCopyLoop = tilingData->lastCopyLoop;
        this->lastCopyTail = tilingData->lastCopyTail;
        this->availableUbSize = tilingData->availableUbSize;
        this->totalResult = tilingData->totalResult;
        this->ptsTotal = tilingData->ptsTotal;
        this->dimSizeSecondLast = tilingData->dimSizeSecondLast;
        this->dimSizeLast = tilingData->dimSizeLast;
        projectionMatGm.SetGlobalBuffer((__gm__ DTYPE_PROJECTION_MAT*)projectionMat, this->totalResult);
        ptsExtendGm.SetGlobalBuffer((__gm__ DTYPE_PTS_EXTEND*)ptsExtend, this->ptsTotal);
        point2dGm.SetGlobalBuffer((__gm__ DTYPE_POINT_2D*)point2d, this->totalResult);
       
        pipe->InitBuffer(projBuf, this->availableUbSize * dimSizeSecondLast * sizeof(DTYPE_PROJECTION_MAT));
        pipe->InitBuffer(ptsBuf, this->availableUbSize * dimSizeSecondLast * sizeof(DTYPE_PTS_EXTEND));
        pipe->InitBuffer(pointsBuf, this->availableUbSize * dimSizeSecondLast * sizeof(DTYPE_POINT_2D));
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();
        uint64_t startAddress = coreId * this->coreData;
        if (coreId >= this->coreUsed) {
            return;
        }
        if (coreId != (this->coreUsed -1)) {
            for (uint32_t i = 0; i < this->copyLoop; i++) {
                uint64_t address = startAddress + i * this->availableUbSize;
                indicesCompute(i, this->availableUbSize, address);
            }
            if (this->copyTail != 0) {
                uint64_t address = startAddress + this->copyLoop * this->availableUbSize;
                indicesCompute(this->copyLoop, this->copyTail, address);
            }
        } else {
            for (uint32_t i = 0; i < this->lastCopyLoop; i++) {
                uint64_t address = startAddress + i * this->availableUbSize;
                indicesCompute(i, this->availableUbSize, address);
            }
            if (this->lastCopyTail != 0) {
                uint64_t address = startAddress + this->lastCopyLoop * this->availableUbSize;
                indicesCompute(this->lastCopyLoop, this->lastCopyTail, address);
            }
        }
    }

private:
    __aicore__ inline void indicesCompute(int32_t progress, int32_t tensorSize, uint64_t address)
    {
        projMatUb = projBuf.Get<DTYPE_PROJECTION_MAT>();
        ptsUb = ptsBuf.Get<DTYPE_PTS_EXTEND>();
        point2dUb = pointsBuf.Get<DTYPE_POINT_2D>();
        DataCopyPadParams proPadParams{true, 0, 4, 0};
        DataCopyParams copyParamsProjUb{1, (uint16_t)(tensorSize * dimSizeSecondLast * sizeof(DTYPE_PROJECTION_MAT)), 0, 0};
        DataCopyPad(ptsUb, ptsExtendGm[address * dimSizeLast], copyParamsProjUb, padParams);
        DataCopyPad(projMatUb, projectionMatGm[address * dimSizeLast], copyParamsProjUb, padParams);
        PipeBarrier<PIPE_ALL>();
        Mul(point2dUb, projMatUb, ptsUb, tensorSize * dimSizeLast);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(point2dGm[address * dimSizeSecondLast], point2dUb, copyParamsProjUb);
        PipeBarrier<PIPE_ALL>();
    }
};

extern "C" __global__ __aicore__ void batch_matmul_vector(GM_ADDR projectionMat, GM_ADDR ptsExtend,
                                                        GM_ADDR point2d,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelBatchMatmulVector op;
    TPipe pipe;
    op.Init(projectionMat, ptsExtend, point2d, workspace, &tilingData, &pipe);
    op.Process();
}