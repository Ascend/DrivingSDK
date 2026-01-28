/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

namespace {
    constexpr uint32_t HALF_BYTE_SIZE = 2;
    constexpr uint32_t INT32_BYTE_SIZE = 4;
    constexpr uint32_t LOCAL_TENSOR_NUM = 4;
}

template <typename T>
class SigmoidFocalLossKernel {
    public:
        __aicore__ inline SigmoidFocalLossKernel() {};

        __aicore__ inline void Init(TPipe *pipe, GM_ADDR logit, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, SigmoidFocalLossTilingData *tilingData) {
            pipe_ = pipe;
            blkIdx_ = GetBlockIdx();
            InitTiling(tilingData);
            InitGM(logit, target, weight, loss);
            InitUB();
            eventMTE2ToV_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
            evnetVToMTE3_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        }

        __aicore__ inline void InitTiling(SigmoidFocalLossTilingData *tilingData) {
            numSamples_ = tilingData->numSamples;
            numClasses_ = tilingData->numClasses;
            numClassesAlign_ = tilingData->numClassesAlign;

            usedCoreNum_ = tilingData->usedCoreNum;
            numHeadCores_ = tilingData->numHeadCores;
            numTailCores_ = tilingData->numTailCores;
            numTaskOnHeadCore_ = tilingData->numTaskOnHeadCore;
            numTaskOnTailCore_ = tilingData->numTaskOnTailCore;

            numLoopOnHeadCore_ = tilingData->numLoopOnHeadCore;
            numTaskPerLoopOnHeadCore_ = tilingData->numTaskPerLoopOnHeadCore;
            numTaskTailOnHeadCore_ = tilingData->numTaskTailOnHeadCore;
            numLoopOnTailCore_ = tilingData->numLoopOnTailCore;
            numTaskPerLoopOnTailCore_ = tilingData->numTaskPerLoopOnTailCore;
            numTaskTailOnTailCore_ = tilingData->numTaskTailOnTailCore;

            gamma_ = tilingData->gamma;
            alpha_ = tilingData->alpha;


            if (blkIdx_ < numHeadCores_) {
                numLoop_ = numLoopOnHeadCore_;
                numTaskPerLoop_ = numTaskPerLoopOnHeadCore_;
                numTaskTail_ = numTaskTailOnHeadCore_;
                globalOffset_ = numTaskOnHeadCore_ * blkIdx_ * numClasses_;
            } else {
                numLoop_ = numLoopOnTailCore_;
                numTaskPerLoop_ = numTaskPerLoopOnTailCore_;
                numTaskTail_ = numTaskTailOnTailCore_;
                globalOffset_ = numTaskOnHeadCore_ * numHeadCores_ * numClasses_ + numTaskOnTailCore_ * (blkIdx_ - numHeadCores_) * numClasses_;
            }
        }

        __aicore__ inline void InitGM(GM_ADDR logit, GM_ADDR target, GM_ADDR weight, GM_ADDR loss) {            
            logitGM_.SetGlobalBuffer((__gm__ T*) logit + globalOffset_);
            targetGM_.SetGlobalBuffer((__gm__ int32_t*) target + globalOffset_);
            weightGM_.SetGlobalBuffer((__gm__ T*) weight + globalOffset_);
            lossGM_.SetGlobalBuffer((__gm__ T*) loss + globalOffset_);        
        }

        __aicore__ inline void InitUB() {
            uint32_t elementCount = numTaskPerLoop_ * numClassesAlign_;
            pipe_->InitBuffer(targetBuf_, numTaskPerLoop_ * numClasses_ * INT32_BYTE_SIZE);
            pipe_->InitBuffer(totalBuf_, LOCAL_TENSOR_NUM * elementCount * sizeof(T));
            pipe_->InitBuffer(tmpBuf_, numTaskPerLoop_ * numClasses_ * sizeof(float));

            targetLocal_ = targetBuf_.Get<int32_t>();

            logitLocal_ = totalBuf_.Get<T>();
            weightLocal_ = logitLocal_[elementCount];
            lossLocal_ = weightLocal_[elementCount];
            targetCastLocal_ = lossLocal_[elementCount];
            tmpLocal_ = tmpBuf_.Get<float>();
            
            isFp16 = (sizeof(T) == HALF_BYTE_SIZE) ? true : false;
        }

        __aicore__ inline void Process() {
            uint32_t dataCount = numTaskPerLoop_ * numClasses_;
            uint16_t oneRepeatSize = GetVecLen() / sizeof(T);
            uint16_t repeatTimes = CeilDivision(dataCount, oneRepeatSize);
            for (uint32_t loop = 0; loop < numLoop_; loop++) {
                DataCopyPad(logitLocal_, logitGM_[loop * dataCount],
                    {1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
                DataCopyPad(targetLocal_, targetGM_[loop * dataCount], 
                    {1, dataCount * INT32_BYTE_SIZE, 0, 0, 0}, {false, 0, 0, 0});
                DataCopyPad(weightLocal_, weightGM_[loop * dataCount], 
                    {1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});

                SetFlag<HardEvent::MTE2_V>(eventMTE2ToV_);
                WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV_);

                if (isFp16) {
                    Cast(tmpLocal_, targetLocal_, RoundMode::CAST_RINT, dataCount);
                    Cast(targetCastLocal_, tmpLocal_, RoundMode::CAST_RINT, dataCount);
                } else {
                    Cast(targetCastLocal_, targetLocal_, RoundMode::CAST_RINT, dataCount);
                }

                PipeBarrier<PIPE_ALL>();
                ComputeVf(dataCount, oneRepeatSize, repeatTimes);
                PipeBarrier<PIPE_ALL>();

                DataCopyPad(lossGM_[loop * dataCount], lossLocal_, {1, static_cast<uint16_t>(dataCount * sizeof(T)), 0, 0});
                
            }
            
            // 处理尾数据
            if (numTaskTail_ > 0) {
                uint32_t dataCount = numTaskTail_ * numClasses_;
                DataCopyPad(logitLocal_, logitGM_[numLoop_ * numTaskPerLoop_ * numClasses_], 
                    {1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
                DataCopyPad(targetLocal_, targetGM_[numLoop_ * numTaskPerLoop_ * numClasses_], 
                    {1, dataCount * INT32_BYTE_SIZE, 0, 0, 0}, {false, 0, 0, 0});
                DataCopyPad(weightLocal_, weightGM_[numLoop_ * numTaskPerLoop_ * numClasses_], 
                    {1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});

                SetFlag<HardEvent::MTE2_V>(eventMTE2ToV_);
                WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV_);

                uint16_t repeatTimes = CeilDivision(dataCount, oneRepeatSize);

                if (isFp16) {
                    Cast(tmpLocal_, targetLocal_, RoundMode::CAST_RINT, dataCount);
                    Cast(targetCastLocal_, tmpLocal_, RoundMode::CAST_RINT, dataCount);
                } else {
                    Cast(targetCastLocal_, targetLocal_, RoundMode::CAST_RINT, dataCount);
                }

                PipeBarrier<PIPE_ALL>();
                ComputeVf(dataCount, oneRepeatSize, repeatTimes);
                PipeBarrier<PIPE_ALL>();

                DataCopyPad(lossGM_[numLoop_ * numTaskPerLoop_ * numClasses_], lossLocal_, {1, static_cast<uint16_t>(dataCount * sizeof(T)), 0, 0});
            }

        }

        __aicore__ inline void ComputeVf(uint32_t dataCount, uint16_t oneRepeatSize, uint16_t repeatTimes) {

            __local_mem__ T* inputPtr = (__local_mem__ T*)logitLocal_.GetPhyAddr();
            __local_mem__ T* targetPtr = (__local_mem__ T*)targetCastLocal_.GetPhyAddr();
            __local_mem__ T* weightPtr = (__local_mem__ T*)weightLocal_.GetPhyAddr();
            __local_mem__ T* lossPtr = (__local_mem__ T*)lossLocal_.GetPhyAddr();

            __VEC_SCOPE__ {
                MicroAPI::RegTensor<T> logitReg;
                MicroAPI::RegTensor<T> pReg;
                MicroAPI::RegTensor<T> targetReg;
                MicroAPI::RegTensor<T> weightReg;
                MicroAPI::RegTensor<T> lossReg;
                MicroAPI::RegTensor<T> oneSubPReg;
                MicroAPI::RegTensor<T> logPReg;
                MicroAPI::RegTensor<T> logOneSubPReg;
                MicroAPI::RegTensor<T> tmp1Reg;
                MicroAPI::RegTensor<T> tmp2Reg;
                MicroAPI::RegTensor<T> positiveReg;
                MicroAPI::RegTensor<T> negativeReg;
                MicroAPI::RegTensor<T> oneSubTargetReg;
                MicroAPI::RegTensor<T> resultReg;

                MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();

                for (uint16_t i = 0; i < repeatTimes; i++) {
                    MicroAPI::DataCopy(logitReg, inputPtr + i * oneRepeatSize);
                    MicroAPI::DataCopy(targetReg, targetPtr + i * oneRepeatSize);
                    MicroAPI::DataCopy(weightReg, weightPtr + i * oneRepeatSize);
                    
                    // 计算p
                    MicroAPI::Neg(tmp1Reg, logitReg, mask);
                    MicroAPI::Exp(tmp1Reg, tmp1Reg, mask);
                    MicroAPI::Duplicate(tmp2Reg, T(1.0), mask);
                    MicroAPI::Add(tmp1Reg, tmp1Reg, tmp2Reg, mask);
                    MicroAPI::Div(pReg, tmp2Reg, tmp1Reg, mask);

                    // 计算1-p
                    MicroAPI::Neg(oneSubPReg, pReg, mask);
                    MicroAPI::Adds(oneSubPReg, oneSubPReg, 1, mask);

                    // 计算log(p)和log(1-p)
                    MicroAPI::Log(logPReg, pReg, mask);
                    MicroAPI::Log(logOneSubPReg, oneSubPReg, mask);
                    
                    // 计算p^gamma=exp(gamma*log(p))和(1-p)^gamma = exp(gamma*log(1-p)) 
                    MicroAPI::Muls(tmp1Reg, logPReg, gamma_, mask);
                    MicroAPI::Muls(tmp2Reg, logOneSubPReg, gamma_, mask);
                    MicroAPI::Exp(tmp1Reg, tmp1Reg, mask);
                    MicroAPI::Exp(tmp2Reg, tmp2Reg, mask);
                    
                    // 计算-alpha*(1-p)^gamma*log(p)和-(1-apha)*(p^gamma)*log(1-p)
                    MicroAPI::Muls(positiveReg, tmp2Reg, -alpha_, mask);
                    MicroAPI::Muls(negativeReg, tmp1Reg, alpha_ - T(1.0), mask);
                    MicroAPI::Mul(positiveReg, positiveReg, logPReg, mask);
                    MicroAPI::Mul(negativeReg, negativeReg, logOneSubPReg, mask);
                    
                    // positiveReg * targetReg + negativeReg * （1 - targetReg)
                    MicroAPI::Neg(oneSubTargetReg, targetReg, mask);
                    MicroAPI::Adds(oneSubTargetReg, oneSubTargetReg, 1, mask);
                    MicroAPI::Mul(resultReg, positiveReg, targetReg, mask);
                    MicroAPI::MulAddDst(resultReg, negativeReg, oneSubTargetReg, mask);

                    MicroAPI::Mul(resultReg, resultReg, weightReg, mask);

                    MicroAPI::DataCopy(lossPtr + i * oneRepeatSize, resultReg, mask);
                }

            }

        }
    
    private:
        TPipe* pipe_;
        uint32_t blkIdx_, numSamples_, numClasses_, numClassesAlign_, usedCoreNum_, numHeadCores_, numTailCores_, numTaskOnHeadCore_, numTaskOnTailCore_, 
            numLoopOnHeadCore_, numTaskPerLoopOnHeadCore_, numTaskTailOnHeadCore_, numLoopOnTailCore_, numTaskPerLoopOnTailCore_, numTaskTailOnTailCore_,
            numLoop_, numTaskPerLoop_, numTaskTail_, globalOffset_;
        float gamma_, alpha_;
        int32_t eventMTE2ToV_, evnetVToMTE3_;
        bool isFp16;
        GlobalTensor<T> logitGM_, weightGM_, lossGM_;
        GlobalTensor<int32_t> targetGM_;
        LocalTensor<T> logitLocal_, weightLocal_, lossLocal_, targetCastLocal_;
        LocalTensor<int32_t> targetLocal_;
        LocalTensor<float> tmpLocal_;
        TBuf<TPosition::VECCALC> targetBuf_, totalBuf_, tmpBuf_;
};

extern "C" __global__ __aicore__ void sigmoid_focal_loss(GM_ADDR logit, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    SigmoidFocalLossKernel<DTYPE_LOGIT> op;
    TPipe pipe;
    op.Init(&pipe, logit, target, weight, loss, &tiling_data);
    op.Process();
}