
#include "simt_api/device_functions.h"
#include "basic_api/kernel_operator_intf.h"
#include "basic_api/kernel_tensor.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;
using namespace MicroAPI;

constexpr int32_t BYTE_SIZE_PER_BLOCK = 32;
constexpr int32_t THREAD_NUM = 1024;

/**
    taskCountPerThread：每个thread处理几个task
    vfTaskCount：该simtVF总共处理几个task
    把任务indices从taskIdx开始，从GM copy taskCount个数据到UB中
    idx_0_k_0 idx_0_k_1 idx_0_k_2 idx_0_k_3 ....
    idx_1_k_0 idx_1_k_1 idx_1_k_2 idx_1_k_3 ....
    ....
    注意点：
    充分利用fp4/fp2/half4/half2等向量类型，提升访存带宽
**/

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void CopyInIndicesOneMapNoKeySimt(
    __gm__ volatile int32_t* indicesOffsetGlobal, __gm__ volatile int4* indicesInputGlobal,
    __gm__ volatile int32_t* map1Global, int32_t blkIdx, int32_t blockNums, int32_t totalTaskCount,
    int32_t spatialShape0, int32_t spatialShape1, int32_t spatialShape2, int32_t k0, int32_t k1, int32_t k2)
{
    int32_t totalSpatialShape = spatialShape0 * spatialShape1 * spatialShape2;
    int32_t spatialShape12 = spatialShape1 * spatialShape2;

    for (uint32_t idx = blkIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); idx < totalTaskCount;
        idx += blockNums * Simt::GetThreadNum()) {

        int32_t bs = indicesInputGlobal[idx].x;
        int32_t sp0 = indicesInputGlobal[idx].y;
        int32_t sp1 = indicesInputGlobal[idx].z;
        int32_t sp2 = indicesInputGlobal[idx].w;

        for (int32_t k0Idx = 0; k0Idx < k0; k0Idx++) {
            for (int32_t k1Idx = 0; k1Idx < k1; k1Idx++) {
                for (int32_t k2Idx = 0; k2Idx < k2; k2Idx++) {
                    int32_t k = k0Idx * k1 * k2 + k1Idx * k2 + k2Idx;
                    int32_t mapOffset = bs * totalSpatialShape + (k0Idx + sp0) * spatialShape12 +
                        (k1Idx + sp1) * spatialShape2 + sp2 + k2Idx;
                    indicesOffsetGlobal[k * totalTaskCount + idx] = map1Global[mapOffset];
                }
            }
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void CopyInIndicesTwoMapNoKeySimt(
    __gm__ volatile int32_t* indicesOffsetGlobal, __gm__ volatile int4* indicesInputGlobal,
    __gm__ volatile int32_t* map1Global, __gm__ volatile int32_t* map2Global, int32_t blkIdx,
    int32_t blockNums, int32_t totalTaskCount, int32_t spatialShape0, int32_t spatialShape1,
    int32_t spatialShape2, int32_t k0, int32_t k1, int32_t k2)
{
    for (uint32_t idx = blkIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); idx < totalTaskCount;
        idx += blockNums * Simt::GetThreadNum()) {

        int32_t bs = indicesInputGlobal[idx].x;
        int32_t sp0 = indicesInputGlobal[idx].y;
        int32_t sp1 = indicesInputGlobal[idx].z;
        int32_t sp2 = indicesInputGlobal[idx].w;

        for (int32_t k0Idx = 0; k0Idx < k0; k0Idx++) {
            for (int32_t k1Idx = 0; k1Idx < k1; k1Idx++) {
                for (int32_t k2Idx = 0; k2Idx < k2; k2Idx++) {
                    int32_t k = k0Idx * k1 * k2 + k1Idx * k2 + k2Idx;
                    int32_t map1Offset = bs * spatialShape0 * spatialShape1 +
                        (sp0 + k0Idx) * spatialShape1 + (sp1 + k1Idx);
                    int32_t map1Val = map1Global[map1Offset];
                    
                    if (map1Val == -1) {
                        indicesOffsetGlobal[k * totalTaskCount + idx] = -1;
                        continue;
                    }

                    int32_t map2Offset = map1Val * spatialShape2 + sp2 + k2Idx;
                    int32_t map2Val = map2Global[map2Offset];
                    indicesOffsetGlobal[k * totalTaskCount + idx] = map2Val;
                }
            }
        }
    }
}


template<typename T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void GatherFeatureVectorDTypeSimt(
    __ubuf__ int32_t* validIndices1Local, __ubuf__ int32_t* validIndices2Local,
    __gm__ volatile U* inputFeatureGlobal, __ubuf__ U* gatheredFeature1Local,
    __ubuf__ U* gatheredFeature2Local, int32_t vectorDtypeCountPerChannel, int32_t vfTaskCount,
    int32_t taskOffset)
{
    for (int32_t i = Simt::GetThreadIdx<1>(); i < vfTaskCount; i += Simt::GetThreadNum<1>()) {
        int32_t mapVal1 = validIndices1Local[i];
        int32_t mapVal2 = validIndices2Local[i] + taskOffset;
        gatheredFeature1Local[i * vectorDtypeCountPerChannel + Simt::GetThreadIdx<0>()] =
            inputFeatureGlobal[(mapVal1 * vectorDtypeCountPerChannel + Simt::GetThreadIdx<0>())];
        gatheredFeature2Local[i * vectorDtypeCountPerChannel + Simt::GetThreadIdx<0>()] =
            inputFeatureGlobal[(mapVal2 * vectorDtypeCountPerChannel + Simt::GetThreadIdx<0>())];
    }
}

template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void GatherFeatureSimt(
    __ubuf__ int32_t* validIndices1Local, __ubuf__ int32_t* validIndices2Local,
    __gm__ volatile T* inputFeatureGlobal, __ubuf__ T* gatheredFeature1Local,
    __ubuf__ T* gatheredFeature2Local, int32_t inChannels, int32_t vfTaskCount, int32_t taskOffset)
{
    int32_t tid = threadIdx.x;
    int32_t elementsCountPerBlock = BYTE_SIZE_PER_BLOCK / sizeof(T);
    int32_t inChannelsAligned = (inChannels + elementsCountPerBlock - 1) / elementsCountPerBlock * elementsCountPerBlock;
    int32_t taskCountTimesChannel = vfTaskCount * inChannels;
    for (int32_t i = tid; i < taskCountTimesChannel; i += THREAD_NUM) {
        int32_t taskIdx = i / inChannels;
        int32_t c = i % inChannels;
        int32_t mapVal1 = validIndices1Local[taskIdx];
        int32_t mapVal2 = validIndices2Local[taskIdx] + taskOffset;
        gatheredFeature1Local[taskIdx * inChannelsAligned + c] = inputFeatureGlobal[mapVal1 * inChannels + c];
        gatheredFeature2Local[taskIdx * inChannelsAligned + c] = inputFeatureGlobal[mapVal2 * inChannels + c];
    }
}

template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ScatterFeatureSimt(
    __ubuf__ int32_t* validIndices1Local, __ubuf__ int32_t* validIndices2Local,
    __gm__ float* outputFeatureGlobal, __ubuf__ float* matmulRes1Local, __ubuf__ float* matmulRes2Local,
    int32_t outChannels, int32_t vfTaskCount, int32_t taskOffset)
{
    int32_t xTid = threadIdx.x;
    int32_t yTid = threadIdx.y;
    int32_t yBlockDim = blockDim.y;

    for (int32_t i = vfTaskCount - yTid - 1; i >= 0; i -= yBlockDim) {
        int32_t c = xTid * 2;
        int32_t mapVal1 = validIndices1Local[i];
        int32_t mapVal2 = validIndices2Local[i] + taskOffset;
        Simt::AtomicAdd(outputFeatureGlobal + mapVal2 * outChannels + c, matmulRes1Local[i * outChannels + c]);
        Simt::AtomicAdd(outputFeatureGlobal + mapVal2 * outChannels + c + 1, matmulRes1Local[i * outChannels + c + 1]);
        Simt::AtomicAdd(outputFeatureGlobal + mapVal1 * outChannels + c, matmulRes2Local[i * outChannels + c]);
        Simt::AtomicAdd(outputFeatureGlobal + mapVal1 * outChannels + c + 1, matmulRes2Local[i * outChannels + c + 1]);
    }
}