#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "simt_api/device_functions.h"
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void PrepareGlobalHashMap(__gm__ int32_t* sortedIndicesPtr,
    __gm__ int32_t* indicesOffsetPtr, __gm__ int32_t* inputIdxWspPtr, __gm__ int32_t* outIdxWspPtr,
    __gm__ uint8_t* kIdxWspPtr, int32_t totalTaskNum, int32_t kernelSize, int32_t startFormIdx)
{
    for (int32_t pointIdx = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); pointIdx < totalTaskNum;
        pointIdx += Simt::GetThreadNum() * Simt::GetBlockNum()) {
        int32_t startIdx = indicesOffsetPtr[pointIdx];
        int32_t endIdx = indicesOffsetPtr[pointIdx + 1];
        for (int32_t formIdx = startIdx; formIdx < endIdx; ++formIdx) {
            int32_t sortedIdx = sortedIndicesPtr[formIdx];
            int32_t inputIdx = sortedIdx / kernelSize;
            uint8_t kIdx = sortedIdx % kernelSize;

            int32_t idx = formIdx - startFormIdx;
            asc_stwt(outIdxWspPtr + idx, pointIdx);
            asc_stwt(inputIdxWspPtr + idx, inputIdx);
            asc_stwt(kIdxWspPtr + idx, kIdx);
        }
    }
}
