#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
#define OFFSET_UNIT 27
#define OFFSET_UNIT_BYTE 108
#define OFFSET_ALIGNED 32
#define OFFSET_ALIGNED_BYTE 128
#define KERNEL_SIZE_ALIGNED 16
#define TWO_TIMES_KERNEL_SIZE_ALIGNED 32
#define KERNEL_SIZE_BYTE 36
#define KERNEL_SIZE 9
#define KH 3
#define KW 3
#define SH 1
#define SW 1
#define PH 1
#define PW 1
constexpr int32_t BUFFER_NUM = 2;

template<typename inT, typename outT>
class KernelDeformableConv2d {
public:
    __aicore__ inline KernelDeformableConv2d() {}
    using inputAType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, inT>;
    using inputBType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, inT, false, LayoutMode::NONE, true>;
    using outputCType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, outT>;
    using biasType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, outT>;

    static constexpr MatmulConfig CFG_MDL = GetMDLConfig(false, false, 2, false, true, false, true);
    matmul::Matmul<inputAType, inputBType, outputCType, biasType, CFG_MDL> matmulObj;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR offset, GM_ADDR weight, GM_ADDR bias, GM_ADDR x_offset, GM_ADDR y,
        const DeformableConv2dTilingData *tiling_data, TPipe* pipe)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->tiling_ = *tiling_data;
        this->singleLoopTask = tiling_.task_single_loop;
        if (GetBlockIdx() == tiling_.use_core_num - 1) {
            this->curCoreTask = tiling_.core_avg_task + tiling_.main_core_num;
        } else {
            this->curCoreTask = tiling_.core_avg_task;
        }
        this->singleCoreTask = tiling_.core_avg_task;
        this->tailLoopTask = curCoreTask % singleLoopTask;
        this->loopCount = curCoreTask / singleLoopTask;
        this->out_feature_map_size = tiling_.h_out * tiling_.w_out;
        this->in_feature_map_size = tiling_.h_in * tiling_.w_in;
        this->core_task_offset = GetBlockIdx() * this->singleCoreTask;
        eventID1 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        eventID2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        xGm.SetGlobalBuffer((__gm__ inT*)x, tiling_.x_size);
        offsetGm.SetGlobalBuffer((__gm__ inT*)offset + this->core_task_offset * OFFSET_UNIT, this->curCoreTask * OFFSET_UNIT);
        xOffsetGm.SetGlobalBuffer((__gm__ inT*)x_offset + this->core_task_offset * tiling_.x_offset_unit, this->curCoreTask * tiling_.x_offset_unit);
        weightGm.SetGlobalBuffer((__gm__ inT*)weight, tiling_.weight_size);
        yGm.SetGlobalBuffer((__gm__ outT*)y + this->core_task_offset * tiling_.c_out, this->curCoreTask * tiling_.c_out);
        pipe->InitBuffer(inQueueOffset, BUFFER_NUM, 3 * KERNEL_SIZE_ALIGNED * singleLoopTask * sizeof(inT)); // aligned
        pipe->InitBuffer(outQueueOffset, BUFFER_NUM, KERNEL_SIZE * tiling_.c_in_aligned * sizeof(inT)); // aligned
        pipe->InitBuffer(xTransBuffer, 2 * KERNEL_SIZE_ALIGNED * sizeof(inT));
        pipe->InitBuffer(xBlockBuffer, KERNEL_SIZE * 4 * tiling_.c_in_aligned * sizeof(inT)); // datacopy 4 datablock in one times
        pipe->InitBuffer(pInitBuffer, TWO_TIMES_KERNEL_SIZE_ALIGNED * sizeof(inT));
        pipe->InitBuffer(pBuffer, TWO_TIMES_KERNEL_SIZE_ALIGNED * sizeof(inT));
        pipe->InitBuffer(pCeilBuffer, TWO_TIMES_KERNEL_SIZE_ALIGNED * sizeof(inT));
        InitCopyParams();
    }

    __aicore__ inline void InitCopyParams()
    {
        copyOutParams = {static_cast<uint16_t>(KERNEL_SIZE), static_cast<uint32_t>(tiling_.c_in * sizeof(inT)), 0, 0, 0};
        uint32_t gap1 = (tiling_.w_in - 2) * tiling_.c_in * sizeof(inT);
        uint32_t gap2 = (tiling_.w_in - 1) * tiling_.c_in * sizeof(inT);
        xCopyParams1 = {2, static_cast<uint32_t>(2 * tiling_.c_in * sizeof(inT)), gap1, 0, 0};
        xCopyParams2 = {1, static_cast<uint32_t>(2 * tiling_.c_in * sizeof(inT)), 0, 0, 0};
        xCopyParams3 = {2, static_cast<uint32_t>(tiling_.c_in * sizeof(inT)), gap2, 0, 0};
        xCopyParams4 = {1, static_cast<uint32_t>(tiling_.c_in * sizeof(inT)), 0, 0, 0};
    }

    __aicore__ inline void Process()
    {
        pInitLocal = pInitBuffer.Get<inT>();
        for (int32_t i = 0; i < KH; i++) {
            for (int32_t j = 0; j < KW; j++) {
                int32_t idx = i * KW + j;
                pInitLocal.SetValue(idx, static_cast<inT>(j - static_cast<int32_t>(PW)));
                pInitLocal.SetValue(KERNEL_SIZE_ALIGNED + idx, static_cast<inT>(i - static_cast<int32_t>(PH)));
            }
        }

        uint32_t xOffsetIdx = 0;
        uint32_t yIdx = 0;
        uint32_t offsetIdx = 0;
        uint32_t xOffsetIdxUnit = singleLoopTask * tiling_.x_offset_unit;
        uint32_t yIdxUnit = singleLoopTask * tiling_.c_out;
        uint32_t offsetIdxUnit = singleLoopTask * OFFSET_UNIT;
        int32_t curLoopTaskIdx = 0;
        int32_t globalTaskIdx = this->core_task_offset;
        
        matmulObj.SetTensorB(weightGm);

        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(offsetIdx, singleLoopTask);
            xOffsetCompute(singleLoopTask, curLoopTaskIdx, globalTaskIdx);
            matmulObj.SetTensorA(xOffsetGm[xOffsetIdx]);
            matmulObj.template IterateAll<false>(yGm[yIdx]);
            xOffsetIdx += xOffsetIdxUnit;
            yIdx += yIdxUnit;
            offsetIdx += offsetIdxUnit;
            curLoopTaskIdx += singleLoopTask;
            globalTaskIdx += singleLoopTask;
        }
        if (tailLoopTask != 0) {
            CopyIn(offsetIdx, tailLoopTask);
            xOffsetCompute(tailLoopTask, curLoopTaskIdx, globalTaskIdx);
            matmulObj.SetTensorA(xOffsetGm[xOffsetIdx]);
            matmulObj.SetTail(tailLoopTask, tiling_.c_out, tiling_.x_offset_unit);
            matmulObj.template IterateAll<false>(yGm[yIdx]);
        }
        matmulObj.End();
    }
private:
    __aicore__ inline void CopyIn(int32_t offsetIdx, int32_t curLoopTask)
    {
        LocalTensor tmpOffsetLocal = inQueueOffset.AllocTensor<inT>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(curLoopTask),
        static_cast<uint32_t>(OFFSET_UNIT_BYTE), 0, 0, 0};
        DataCopyPad(tmpOffsetLocal, offsetGm[offsetIdx], copyParams, padParams); // 非对齐
        inQueueOffset.EnQue(tmpOffsetLocal);
    }
    __aicore__ inline void xOffsetCompute(int32_t curLoopTask, int32_t curLoopTaskIdx, int32_t globalTaskIdx)
    {
        pLocal = pBuffer.Get<inT>();
        pCeilLocal = pCeilBuffer.Get<inT>();
        xblockLocal = xBlockBuffer.Get<inT>();
        xtransLocal = xTransBuffer.Get<inT>();
        offsetLocal = inQueueOffset.DeQue<inT>();
        for (int32_t i = 0; i < curLoopTask; i++) {
            xOffsetLocal = outQueueOffset.AllocTensor<inT>();
            uint32_t taskIdx = globalTaskIdx + i;
            int32_t h_in_lt = (taskIdx % out_feature_map_size / tiling_.w_out) * SH;  // kernel left top index at H_in axis
            int32_t w_in_lt = (taskIdx % tiling_.w_out) * SW;  // kernel left top index at W_in axis
            uint32_t offsetLocalIdx = OFFSET_ALIGNED * i;
            GatherMask(xtransLocal, offsetLocal[offsetLocalIdx], src1Pattern1, false,
                mask, {1, 1, 0, 0}, rsvdCnt);
            GatherMask(xtransLocal[KERNEL_SIZE_ALIGNED], offsetLocal[offsetLocalIdx], src1Pattern2, false,
                mask, {1, 1, 0, 0}, rsvdCnt);
            Adds(pLocal, pInitLocal, static_cast<inT>(w_in_lt), KERNEL_SIZE);
            Adds(pLocal[KERNEL_SIZE_ALIGNED], pInitLocal[KERNEL_SIZE_ALIGNED], static_cast<inT>(h_in_lt), static_cast<uint32_t>(KERNEL_SIZE));
            Add(pLocal, pLocal, xtransLocal, TWO_TIMES_KERNEL_SIZE_ALIGNED);
            AscendC::PipeBarrier<PIPE_V>();
            Ceil(pCeilLocal, pLocal, TWO_TIMES_KERNEL_SIZE_ALIGNED);
            AscendC::PipeBarrier<PIPE_V>();
            Sub(pLocal, pCeilLocal, pLocal, TWO_TIMES_KERNEL_SIZE_ALIGNED);
            Duplicate(xOffsetLocal, static_cast<inT>(0.0), KERNEL_SIZE * tiling_.c_in_aligned);
            int32_t b_idx = taskIdx / out_feature_map_size;
            uint32_t curOffset = 0;
            for (int32_t j = 0; j < KERNEL_SIZE; j++) {
                BilinearInterpolate(j, b_idx, curOffset, offsetLocalIdx);
                curOffset += tiling_.c_in_aligned;
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventID2);
            DataCopyPad(xOffsetGm[(curLoopTaskIdx + i) * tiling_.x_offset_unit], xOffsetLocal, copyOutParams);
            outQueueOffset.FreeTensor(xOffsetLocal);
        }
        inQueueOffset.FreeTensor(offsetLocal);
    }
    __aicore__ inline void BilinearInterpolate(int32_t kidx, int32_t b_idx, int32_t x_offset_idx, int32_t offsetLocalIdx)
    {
        int32_t wceil = pCeilLocal.GetValue(kidx);
        int32_t hceil = pCeilLocal.GetValue(KERNEL_SIZE_ALIGNED + kidx);
        float weight_w, weight_h;
        weight_w = pLocal.GetValue(kidx);
        weight_h = pLocal.GetValue(KERNEL_SIZE_ALIGNED + kidx);
        int32_t idx_lt = b_idx * in_feature_map_size + (hceil - 1) * tiling_.w_in + (wceil - 1);
        uint32_t xBlockLocalOffset = kidx * 4 * tiling_.c_in;
        if (0 < wceil && wceil < tiling_.w_in) {
            if (0 < hceil && hceil < tiling_.h_in) {
                uint32_t x_idx = idx_lt * tiling_.c_in;
                DataCopyPad(xblockLocal[xBlockLocalOffset], xGm[x_idx], xCopyParams1, padParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventID1);
                Muls(xOffsetLocal[x_offset_idx], xblockLocal[xBlockLocalOffset], static_cast<inT>(weight_w * weight_h), tiling_.c_in);
                Axpy(xOffsetLocal[x_offset_idx], xblockLocal[xBlockLocalOffset + tiling_.c_in], static_cast<inT>((1 - weight_w) * weight_h), tiling_.c_in);
                Axpy(xOffsetLocal[x_offset_idx], xblockLocal[xBlockLocalOffset + 2 * tiling_.c_in], static_cast<inT>(weight_w * (1 - weight_h)), tiling_.c_in);
                Axpy(xOffsetLocal[x_offset_idx], xblockLocal[xBlockLocalOffset + 3 * tiling_.c_in], static_cast<inT>((1 - weight_w) * (1 - weight_h)), tiling_.c_in);
            } else if (hceil == 0) {
                uint32_t x_idx = (idx_lt + tiling_.w_in) * tiling_.c_in;
                DataCopyPad(xblockLocal[xBlockLocalOffset], xGm[x_idx], xCopyParams2, padParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventID1);
                Muls(xOffsetLocal[x_offset_idx], xblockLocal[xBlockLocalOffset], static_cast<inT>(weight_w * (1 - weight_h)), tiling_.c_in);
                Axpy(xOffsetLocal[x_offset_idx], xblockLocal[xBlockLocalOffset + tiling_.c_in], static_cast<inT>((1 - weight_w) * (1 - weight_h)), tiling_.c_in);
            } else if (hceil == tiling_.h_in) {
                uint32_t x_idx = idx_lt * tiling_.c_in;
                DataCopyPad(xblockLocal[xBlockLocalOffset], xGm[x_idx], xCopyParams2, padParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventID1);
                Muls(xOffsetLocal[x_offset_idx], xblockLocal[xBlockLocalOffset], static_cast<inT>(weight_w * weight_h), tiling_.c_in);
                Axpy(xOffsetLocal[x_offset_idx], xblockLocal[xBlockLocalOffset + tiling_.c_in], static_cast<inT>((1 - weight_w) * weight_h), tiling_.c_in);
            }
        } else if (wceil == 0) {
            if (0 < hceil && hceil < tiling_.h_in) {
                uint32_t x_idx = (idx_lt + 1) * tiling_.c_in;
                DataCopyPad(xblockLocal[xBlockLocalOffset], xGm[x_idx], xCopyParams3, padParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventID1);
                Muls(xOffsetLocal[x_offset_idx], xblockLocal[xBlockLocalOffset], static_cast<inT>((1 - weight_w) * weight_h), tiling_.c_in);
                Axpy(xOffsetLocal[x_offset_idx], xblockLocal[xBlockLocalOffset + tiling_.c_in], static_cast<inT>((1 - weight_w) * (1 - weight_h)), tiling_.c_in);
            } else if (hceil == 0) {
                uint32_t x_idx = (idx_lt + tiling_.w_in + 1) * tiling_.c_in;
                DataCopyPad(xblockLocal[xBlockLocalOffset], xGm[x_idx], xCopyParams4, padParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventID1);
                Muls(xOffsetLocal[x_offset_idx], xblockLocal[xBlockLocalOffset], static_cast<inT>((1 - weight_w) * (1 - weight_h)), tiling_.c_in);
            } else if (hceil == tiling_.h_in) {
                uint32_t x_idx = (idx_lt + 1) * tiling_.c_in;
                DataCopyPad(xblockLocal[xBlockLocalOffset], xGm[x_idx], xCopyParams4, padParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventID1);
                Muls(xOffsetLocal[x_offset_idx], xblockLocal[xBlockLocalOffset], static_cast<inT>((1 - weight_w) * weight_h), tiling_.c_in);
            }
        } else if (wceil == tiling_.w_in) {
            if (0 < hceil && hceil < tiling_.h_in) {
                uint32_t x_idx = idx_lt * tiling_.c_in;
                DataCopyPad(xblockLocal[xBlockLocalOffset], xGm[x_idx], xCopyParams3, padParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventID1);
                Muls(xOffsetLocal[x_offset_idx], xblockLocal[xBlockLocalOffset], static_cast<inT>(weight_w * weight_h), tiling_.c_in);
                Axpy(xOffsetLocal[x_offset_idx], xblockLocal[xBlockLocalOffset + tiling_.c_in], static_cast<inT>(weight_w * (1 - weight_h)), tiling_.c_in);
            } else if (hceil == 0) {
                uint32_t x_idx = (idx_lt + tiling_.w_in) * tiling_.c_in;
                DataCopyPad(xblockLocal[xBlockLocalOffset], xGm[x_idx], xCopyParams4, padParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventID1);
                Muls(xOffsetLocal[x_offset_idx], xblockLocal[xBlockLocalOffset], static_cast<inT>(weight_w * (1 - weight_h)), tiling_.c_in);
            } else if (hceil == tiling_.h_in) {
                uint32_t x_idx = idx_lt * tiling_.c_in;
                DataCopyPad(xblockLocal[xBlockLocalOffset], xGm[x_idx], xCopyParams4, padParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventID1);
                Muls(xOffsetLocal[x_offset_idx], xblockLocal[xBlockLocalOffset], static_cast<inT>(weight_w * weight_h), tiling_.c_in);
            }
        }
        inT scale_value = offsetLocal.GetValue(2 * KERNEL_SIZE + kidx + offsetLocalIdx);
        Muls(xOffsetLocal[x_offset_idx], xOffsetLocal[x_offset_idx], static_cast<inT>(scale_value), tiling_.c_in);
    }
private:
    TBuf<TPosition::VECCALC> xBlockBuffer, xTransBuffer, pInitBuffer, pBuffer, pCeilBuffer;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueOffset;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOffset;
    GlobalTensor<inT> xGm, offsetGm, xOffsetGm, weightGm, yGm;
    LocalTensor<inT> xblockLocal, offsetLocal, pInitLocal, pLocal, pCeilLocal, xtransLocal, xOffsetLocal;
    DeformableConv2dTilingData tiling_;
    uint32_t singleLoopTask;
    uint32_t singleCoreTask;
    uint32_t core_task_offset;
    uint32_t mask = 0; // normal模式下mask需要设置为0
    uint64_t rsvdCnt = KH * KW;; // 用于保存筛选后保留下来的元素个数
    uint8_t src1Pattern1 = 2; // 偶数索引元素
    uint8_t src1Pattern2 = 1;
    int32_t out_feature_map_size, in_feature_map_size;
    int32_t eventID1, eventID2;
    uint32_t loopCount, tailLoopTask, curCoreTask;
    DataCopyPadExtParams<inT> padParams{false, 0, 0, 0};
    DataCopyExtParams copyOutParams, xCopyParams1, xCopyParams2, xCopyParams3, xCopyParams4;
};
extern "C" __global__ __aicore__ void deformable_conv2d(GM_ADDR x, GM_ADDR offset, GM_ADDR weight, GM_ADDR bias, GM_ADDR x_offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    SetSysWorkspace(workspace);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    KernelDeformableConv2d<DTYPE_X, DTYPE_X> op;
    TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.matmulObj, &tiling_data.cubeTilingData);
    op.Init(x, offset, weight, bias, x_offset, y, &tiling_data, &pipe);
    op.Process();
}
#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void deformable_conv2d_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *y, uint8_t *workspace, uint8_t *tiling)
{
    deformable_conv2d<<<blockDim, l2ctrl, stream>>>(x, offset, weight, bias, x_offset, y, workspace, tiling);
}
#endif