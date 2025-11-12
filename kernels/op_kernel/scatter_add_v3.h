/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#ifndef _SCATTER_ADD_V3_H_
#define _SCATTER_ADD_V3_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

constexpr uint64_t MAX_MASK = 64;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t MASK_ALIGN_SIZE = 256;
constexpr uint64_t DIM_SIZE_THRESHOLD = 200;
constexpr uint64_t MAX_COPY_PAD =  4095;
constexpr uint64_t BUFFER_NUM_MAX = 8;
constexpr uint64_t UB_SIZE_COEFF = 2;

class ScatterAddBaseKernel {
public:
    __aicore__ inline ScatterAddBaseKernel() = delete;

    __aicore__ inline ScatterAddBaseKernel(GM_ADDR src, GM_ADDR indices, GM_ADDR out, 
        ScatterAddTilingDataV3* tiling_data, TPipe* pipe) : _blockIdx(GetBlockIdx()), _pipe(pipe)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        initTiling(tiling_data);
        initGM(src, indices, out);
    }

private:
    __aicore__ inline void initTiling(ScatterAddTilingDataV3* tiling_data) 
    {
        _totalHead = tiling_data->totalHead;
        _tailLen = tiling_data->tailLen;
        _dimSize = tiling_data->dimSize;
        _ubSize = tiling_data->ubSize;
        _srcDimSize = tiling_data->srcDimSize;
        _totalSrcNum = tiling_data->totalSrcNum;
        _totalOutNum = tiling_data->totalOutNum;
        _totalIndicesNum = tiling_data->totalIndicesNum;
        _outNumPerHead= tiling_data->outNumPerHead;
        _indicesNumPerHead = tiling_data->indicesNumPerHead;
        _dataNumPerBlock = BLOCK_SIZE / sizeof(DTYPE_OUT);
    }

    __aicore__ inline void initGM(GM_ADDR src, GM_ADDR indices, GM_ADDR out)
    {
        _srcGm.SetGlobalBuffer((__gm__ DTYPE_SRC*)src, _totalSrcNum);
        _outGm.SetGlobalBuffer((__gm__ DTYPE_OUT*)out, _totalOutNum);
        _indicesGm.SetGlobalBuffer((__gm__ DTYPE_INDICES*)indices, _totalIndicesNum);
    }

protected:
    template <typename T1, typename T2>
    __aicore__ inline T1 ceilDiv(T1 a, T2 b) {
        return b == 0 ? 0 : (a + b - 1) / b;
    };

protected:
    TPipe* _pipe;

    TBuf<TPosition::VECCALC> _srcBuf;
    TBuf<TPosition::VECCALC> _outBuf;
    TBuf<TPosition::VECCALC> _indicesBuf;

    LocalTensor<DTYPE_SRC> _srcLocal;
    LocalTensor<DTYPE_OUT> _outLocal;
    LocalTensor<DTYPE_INDICES> _indicesLocal;

    GlobalTensor<DTYPE_SRC> _srcGm;
    GlobalTensor<DTYPE_OUT> _outGm;
    GlobalTensor<DTYPE_INDICES> _indicesGm;

    uint64_t _blockIdx;
    uint64_t _totalHead;
    uint64_t _tailLen;
    uint64_t _dimSize;
    uint64_t _srcDimSize;
    uint64_t _ubSize;
    uint64_t _totalSrcNum;
    uint64_t _totalOutNum;
    uint64_t _totalIndicesNum;
    uint64_t _outNumPerHead;
    uint64_t _indicesNumPerHead;
    uint64_t _dataNumPerBlock;
    uint64_t _loop;
};

class ScatterAddFullyLoad final : public ScatterAddBaseKernel {
public:
    __aicore__ inline ScatterAddFullyLoad() = delete;

    __aicore__ inline ScatterAddFullyLoad(GM_ADDR src, GM_ADDR indices, 
        GM_ADDR out, ScatterAddTilingDataV3* tiling_data, TPipe* pipe) 
        : ScatterAddBaseKernel(src, indices, out, tiling_data, pipe)
    {
        initTiling(tiling_data);
        initBuffer();
    }

    __aicore__ inline void Process()
    {   
        // step 1: fully load OUT into ub
        _outLocal = _outBuf.Get<DTYPE_OUT>();
        Duplicate(_outLocal, 0.0f, AlignUp(_totalOutNum, BLOCK_SIZE / sizeof(DTYPE_OUT)));

        // step 2: copy SRC and INDICES in batches then compute
        for (uint64_t i = 0; i < _loop; i++) {
            if (_dimSize <= DIM_SIZE_THRESHOLD) {
                computeWithSmallDimSize(i);
            } else {
                computeWithLargeDimSize(i);
            }
        }

        // step 3: copy out
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        SetAtomicAdd<float>();
        DataCopyPad(_outGm, _outLocal, {1, static_cast<uint32_t>(_totalOutNum * sizeof(DTYPE_OUT)), 0, 0, 0});
        SetAtomicNone();
    }

private:
    __aicore__ inline void initTiling(ScatterAddTilingDataV3* tiling_data)
    {
        _indicesNumBigCore = tiling_data->indicesNumBigCore;
        _indicesNumSmallCore = tiling_data->indicesNumSmallCore;
        _bigCoreNum = tiling_data->bigCoreNum;

        if (_blockIdx < _bigCoreNum) {
            _indicesMaxLoadableNum = tiling_data->indicesMaxLoadableNumBigCore;
            _loop = ceilDiv(_indicesNumBigCore, _indicesMaxLoadableNum);
            _indicesNumLeftover = _indicesNumBigCore % _indicesMaxLoadableNum;
            _baseOffset = _blockIdx * _indicesNumBigCore;
        } else {
            _indicesMaxLoadableNum = tiling_data->indicesMaxLoadableNumSmallCore;
            _loop = ceilDiv(_indicesNumSmallCore, _indicesMaxLoadableNum);
            _indicesNumLeftover = _indicesNumSmallCore % _indicesMaxLoadableNum;
            _baseOffset = _bigCoreNum * _indicesNumBigCore + (_blockIdx - _bigCoreNum) * _indicesNumSmallCore;
        }
    }

    __aicore__ inline void initBuffer()
    {
        _pipe->InitBuffer(_srcBuf, AlignUp(_indicesMaxLoadableNum * sizeof(DTYPE_SRC), MASK_ALIGN_SIZE));
        _pipe->InitBuffer(_outBuf, AlignUp(_totalOutNum * sizeof(DTYPE_OUT), MASK_ALIGN_SIZE));
        _pipe->InitBuffer(_indicesBuf, AlignUp(_indicesMaxLoadableNum * sizeof(DTYPE_INDICES), MASK_ALIGN_SIZE));

        // vec operation requires additional memory in this case
        if (_dimSize <= DIM_SIZE_THRESHOLD) {
            uint64_t srcMaskNum = AlignUp(_indicesMaxLoadableNum * sizeof(DTYPE_SRC), MASK_ALIGN_SIZE) / sizeof(DTYPE_SRC);
            uint64_t maskBitNum = AscendCUtils::GetBitSize(sizeof(uint8_t));
            uint64_t maskBufSize = ceilDiv(srcMaskNum, maskBitNum) * sizeof(uint8_t);

            _pipe->InitBuffer(_maskBuf, maskBufSize);
            _pipe->InitBuffer(_selectedSrcBuf, AlignUp(_indicesMaxLoadableNum * sizeof(DTYPE_SRC), MASK_ALIGN_SIZE));
            _pipe->InitBuffer(_reduceSumBuf, BLOCK_SIZE);
        }
    }

    __aicore__ inline void computeWithSmallDimSize(uint64_t i)
    {
        _srcLocal = _srcBuf.Get<DTYPE_SRC>();
        _indicesLocal = _indicesBuf.Get<DTYPE_INDICES>();
        _maskLocal = _maskBuf.Get<uint8_t>();
        _reduceSumLocal = _reduceSumBuf.Get<DTYPE_SRC>();
        _selectedSrcLocal = _selectedSrcBuf.Get<DTYPE_SRC>();

        uint64_t indicesGlobalOffset = _baseOffset + i * _indicesMaxLoadableNum;
        uint64_t loadableNum = calcIndicesLoadableNum(i);

        uint64_t startHeadID = indicesGlobalOffset / _indicesNumPerHead;
        uint64_t endHeadID = (indicesGlobalOffset + loadableNum - 1) / _indicesNumPerHead;

        for (uint64_t headID = startHeadID; headID <= endHeadID; headID++) {
            uint64_t loadOffset = max(indicesGlobalOffset, headID * _indicesNumPerHead);
            uint64_t loadLen = min(indicesGlobalOffset + loadableNum, (headID + 1) * _indicesNumPerHead) - loadOffset;
            uint64_t copyLenAlign32 = AlignUp(loadLen, BLOCK_SIZE / sizeof(DTYPE_SRC));
            uint64_t cmpLenAlign256 = AlignUp(loadLen, MASK_ALIGN_SIZE / sizeof(DTYPE_INDICES));
            uint32_t shape[] = {1, static_cast<uint32_t>(loadLen)};

            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            DataCopy(_srcLocal, _srcGm[loadOffset], copyLenAlign32);
            DataCopy(_indicesLocal, _indicesGm[loadOffset], copyLenAlign32);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            for (uint64_t idxVal = 0; idxVal < _dimSize; idxVal++) {
                CompareScalar(_maskLocal, _indicesLocal, static_cast<DTYPE_INDICES>(idxVal), CMPMODE::EQ, cmpLenAlign256);
                Select(_selectedSrcLocal, _maskLocal, _srcLocal, static_cast<DTYPE_SRC>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, loadLen);
                ReduceSum<DTYPE_SRC, Pattern::Reduce::AR, true>(_reduceSumLocal, _selectedSrcLocal, shape, true);
                uint64_t outOffset = headID * _outNumPerHead + idxVal;
                _outLocal.SetValue(outOffset, _outLocal.GetValue(outOffset) + _reduceSumLocal.GetValue(0));
            }
        }
    }

    __aicore__ inline void computeWithLargeDimSize(uint64_t i)
    {
        uint64_t indicesGlobalOffset = _baseOffset + i * _indicesMaxLoadableNum;
        uint64_t loadableNum = calcIndicesLoadableNum(i);
        uint64_t copyLenAlign32 = AlignUp(loadableNum, BLOCK_SIZE / sizeof(DTYPE_SRC));

        _srcLocal = _srcBuf.Get<DTYPE_SRC>();
        _indicesLocal = _indicesBuf.Get<DTYPE_INDICES>();

        DataCopy(_srcLocal, _srcGm[indicesGlobalOffset], copyLenAlign32);
        DataCopy(_indicesLocal, _indicesGm[indicesGlobalOffset], copyLenAlign32);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        uint64_t startHeadID = indicesGlobalOffset / _indicesNumPerHead;
        uint64_t endHeadID = (indicesGlobalOffset + loadableNum - 1) / _indicesNumPerHead;

        for (uint64_t headID = startHeadID; headID <= endHeadID; headID++) {
            uint64_t localStart = max(indicesGlobalOffset, headID * _indicesNumPerHead) - indicesGlobalOffset;
            uint64_t localEnd = min(indicesGlobalOffset + loadableNum, (headID + 1) * _indicesNumPerHead) - indicesGlobalOffset;

            for (uint64_t pos = localStart; pos < localEnd; pos++) {
                DTYPE_INDICES idxVal = _indicesLocal.GetValue(pos);
                uint64_t outOffset = headID * _outNumPerHead + idxVal;
                _outLocal.SetValue(outOffset, _outLocal.GetValue(outOffset) + _srcLocal.GetValue(pos));
            }
        }
    }

    __aicore__ inline uint64_t calcIndicesLoadableNum(uint64_t i)
    {
        if (_indicesNumLeftover == 0) {
            return _indicesMaxLoadableNum;
        }
        return i == _loop-1 ? _indicesNumLeftover : _indicesMaxLoadableNum;
    }

private:
    TBuf<TPosition::VECCALC> _maskBuf;
    TBuf<TPosition::VECCALC> _selectedSrcBuf;
    TBuf<TPosition::VECCALC> _reduceSumBuf;

    LocalTensor<uint8_t> _maskLocal;
    LocalTensor<DTYPE_SRC> _selectedSrcLocal;
    LocalTensor<DTYPE_SRC> _reduceSumLocal;

    // acquired from tiling data
    uint64_t _indicesNumBigCore;
    uint64_t _indicesNumSmallCore;
    uint64_t _bigCoreNum;
    // calculated
    uint64_t _indicesMaxLoadableNum;
    uint64_t _indicesNumLeftover;
    uint64_t _baseOffset;
};

class ScatterAddMultiHeads final : public ScatterAddBaseKernel {
public:
    __aicore__ inline ScatterAddMultiHeads() = delete;

    __aicore__ inline ScatterAddMultiHeads(GM_ADDR src, GM_ADDR indices, 
        GM_ADDR out, ScatterAddTilingDataV3* tiling_data, TPipe* pipe) 
        : ScatterAddBaseKernel(src, indices, out, tiling_data, pipe)
    {
        initTiling(tiling_data);
        initBuffer();
    }

    __aicore__ inline void Process()
    {
        for (uint64_t i = 0; i < _loop; i++) {
            compute(i);
        }
    }

private:
    __aicore__ inline void initTiling(ScatterAddTilingDataV3* tiling_data)
    {
        _headNumBigCore = tiling_data->headNumBigCore;
        _headNumSmallCore = tiling_data->headNumSmallCore;
        _bigCoreNum = tiling_data->bigCoreNum;
        _headNumPerTask = tiling_data->headNumPerTask;

        if (_blockIdx < _bigCoreNum) {
            _loop = ceilDiv(_headNumBigCore, _headNumPerTask);
            _headNumLeftover = _headNumBigCore % _headNumPerTask;
            _headIdOffset =  _blockIdx * _headNumBigCore;
        } else {
            _loop = ceilDiv(_headNumSmallCore, _headNumPerTask);
            _headNumLeftover = _headNumSmallCore % _headNumPerTask;
            _headIdOffset =  _bigCoreNum * _headNumBigCore + (_blockIdx - _bigCoreNum) * _headNumSmallCore;
        }
    }

    __aicore__ inline void initBuffer()
    {
        uint64_t outMaxLoadNum = AlignUp(_headNumPerTask * _outNumPerHead, _dataNumPerBlock);
        uint64_t indicesMaxLoadNum = AlignUp(_headNumPerTask * _indicesNumPerHead, _dataNumPerBlock);

        _pipe->InitBuffer(_srcBuf, indicesMaxLoadNum * sizeof(DTYPE_SRC));
        _pipe->InitBuffer(_outBuf, outMaxLoadNum * sizeof(DTYPE_OUT));
        _pipe->InitBuffer(_indicesBuf, indicesMaxLoadNum * sizeof(DTYPE_INDICES));
    }

    __aicore__ inline void compute(uint64_t i)
    {
        _srcLocal = _srcBuf.Get<DTYPE_SRC>();
        _outLocal = _outBuf.Get<DTYPE_OUT>();
        _indicesLocal = _indicesBuf.Get<DTYPE_INDICES>();
        
        uint64_t startHeadID = _headIdOffset + i * _headNumPerTask;
        uint64_t headNum = calcHeadNum(i);
        uint64_t outLoadOffset = startHeadID * _outNumPerHead;
        uint64_t indicesLoadOffset = startHeadID * _indicesNumPerHead;
        uint64_t outLoadNum = headNum * _outNumPerHead;
        uint64_t indicesLoadNum = headNum * _indicesNumPerHead;

        pipe_barrier(PIPE_ALL);
        DataCopy(_outLocal, _outGm[outLoadOffset], AlignUp(outLoadNum, _dataNumPerBlock));
        DataCopy(_srcLocal, _srcGm[indicesLoadOffset], AlignUp(indicesLoadNum, _dataNumPerBlock));
        DataCopy(_indicesLocal, _indicesGm[indicesLoadOffset], AlignUp(indicesLoadNum, _dataNumPerBlock));

        for (uint64_t h = 0; h < headNum; h++) {
            uint64_t indicesBaseOffset = h * _indicesNumPerHead;
            uint64_t outBaseOffset = h * _outNumPerHead;

            for (uint64_t k = 0; k < _indicesNumPerHead; k++) {
                uint64_t indicesOffset = indicesBaseOffset + k;
                DTYPE_INDICES idxVal = _indicesLocal.GetValue(indicesOffset);
                int64_t outOffset = outBaseOffset + idxVal;
                _outLocal.SetValue(outOffset, _outLocal.GetValue(outOffset) + _srcLocal.GetValue(indicesOffset));
            }
        }

        DataCopyPad(_outGm[outLoadOffset], _outLocal, {1, static_cast<uint32_t>(outLoadNum*sizeof(DTYPE_OUT)), 0, 0, 0});
    }

    __aicore__ inline uint64_t calcHeadNum(uint64_t i)
    {   
        if (_headNumLeftover == 0) {
            return _headNumPerTask;
        }
        return i == _loop - 1 ? _headNumLeftover : _headNumPerTask;
    }

private:
    uint64_t _headNumBigCore;
    uint64_t _headNumSmallCore;
    uint64_t _bigCoreNum;
    uint64_t _headNumPerTask;
    uint64_t _headNumLeftover;
    uint64_t _headIdOffset;
};

template<bool largeHead>
class ScatterAddHeadInBatch final : public ScatterAddBaseKernel {
public:
    __aicore__ inline ScatterAddHeadInBatch() = delete;

    __aicore__ inline ScatterAddHeadInBatch(GM_ADDR src, GM_ADDR indices, 
        GM_ADDR out, ScatterAddTilingDataV3* tiling_data, TPipe* pipe)
        : ScatterAddBaseKernel(src, indices, out, tiling_data, pipe)
    {
        if constexpr (largeHead) {
            initTilingLargeHead(tiling_data);
        } else {
            initTilingFewHeads(tiling_data);
        }
        initBuffer();
    }

    __aicore__ inline void Process()
    {
        for (uint64_t i = 0; i < _loop; i++) {
            compute(i);
        }
    }

private:
    __aicore__ inline void initTilingLargeHead(ScatterAddTilingDataV3* tiling_data)
    {
        _headNumBigCore = tiling_data->headNumBigCore;
        _headNumSmallCore = tiling_data->headNumSmallCore;
        _bigCoreNum = tiling_data->bigCoreNum;
        _indicesNumPerBatch = tiling_data->indicesNumPerBatch;
        _maxOutNumPerBatch = tiling_data->maxOutNumPerBatch;

        if (_blockIdx < _bigCoreNum) {
            _loop = _headNumBigCore;
            _headIdOffset = _blockIdx * _headNumBigCore;
        } else {
            _loop = _headNumSmallCore;
            _headIdOffset = _bigCoreNum * _headNumBigCore + (_blockIdx - _bigCoreNum) * _headNumSmallCore;
        }

        if (_dimSize <= _maxOutNumPerBatch) {
            _outLoop = 1;
            _outNumPerBatch = _dimSize;
            _outNumLeftover = 0;
        } else {
            _outLoop = ceilDiv(_dimSize, _maxOutNumPerBatch);
            _outNumPerBatch = _maxOutNumPerBatch;
            _outNumLeftover = _dimSize % _maxOutNumPerBatch;
        }

        _indicesLoop = ceilDiv(_indicesNumPerHead, _indicesNumPerBatch);
        _indicesNumLeftover = _indicesNumPerHead % _indicesNumPerBatch;
    }

    __aicore__ inline void initTilingFewHeads(ScatterAddTilingDataV3* tiling_data)
    {
        _indicesNumPerBatch = tiling_data->indicesNumPerBatch;
        _maxOutNumPerBatch = tiling_data->maxOutNumPerBatch;
        _coreNumPerHead = tiling_data->coreNumPerHead;
        _outNumPerCore = tiling_data->outNumPerCore;
        _outNumPartOffset = _outNumPerCore;

        _loop = 1;
        _headIdOffset = _blockIdx / _coreNumPerHead;
        _headPartId = _blockIdx % _coreNumPerHead;

        if (_headPartId == _coreNumPerHead-1) {
            _outNumPerCore = _dimSize - _outNumPerCore * (_coreNumPerHead - 1);
        }

        if (_outNumPerCore <= _maxOutNumPerBatch) {
            _outLoop = 1;
            _outNumPerBatch = _outNumPerCore;
            _outNumLeftover = _outNumPerCore;
        } else {
            _outLoop = ceilDiv(_outNumPerCore, _maxOutNumPerBatch);
            _outNumPerBatch = _maxOutNumPerBatch;
            _outNumLeftover = _outNumPerCore % _maxOutNumPerBatch;
        }

        _indicesLoop = ceilDiv(_indicesNumPerHead, _indicesNumPerBatch);
        _indicesNumLeftover = _indicesNumPerHead % _indicesNumPerBatch;
    }

    __aicore__ inline void initBuffer()
    {
        _pipe->InitBuffer(_srcBuf, AlignUp(_indicesNumPerBatch, _dataNumPerBlock) * sizeof(DTYPE_SRC));
        _pipe->InitBuffer(_outBuf, AlignUp(_outNumPerBatch, MAX_MASK) * sizeof(DTYPE_OUT));
        _pipe->InitBuffer(_indicesBuf, AlignUp(_indicesNumPerBatch, _dataNumPerBlock) * sizeof(DTYPE_INDICES));

        uint64_t maskBitNum = AscendCUtils::GetBitSize(sizeof(uint8_t));
        uint64_t indicesMaskNum = AlignUp(_indicesNumPerBatch * sizeof(DTYPE_INDICES), MASK_ALIGN_SIZE) / sizeof(DTYPE_INDICES);
        _maskLen = AlignUp(ceilDiv(indicesMaskNum, maskBitNum), BLOCK_SIZE);
        _pipe->InitBuffer(_maskBuf, _maskLen * sizeof(uint8_t) * UB_SIZE_COEFF);
    }

    __aicore__ inline void compute(uint64_t i)
    {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

        uint64_t headID = _headIdOffset + i;
        // process one head in batches
        for (uint64_t k = 0; k < _outLoop; k++) {
            uint64_t outNum = calcBatchOutNum(k);
            computeHeadInBatches(k, headID, outNum);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void computeHeadInBatches(uint64_t k, uint64_t headID, uint64_t outNum)
    {
        _srcLocal = _srcBuf.Get<DTYPE_SRC>();
        _outLocal = _outBuf.Get<DTYPE_OUT>();
        _indicesLocal = _indicesBuf.Get<DTYPE_INDICES>();

        uint64_t outHeadOffset = headID * _outNumPerHead;
        uint64_t outLoadOffset;
        if constexpr (largeHead) {
            outLoadOffset = outHeadOffset + k * _outNumPerBatch;
        } else {
            outLoadOffset = outHeadOffset + _headPartId * _outNumPartOffset + k * _outNumPerBatch;
        }

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        DataCopy(_outLocal, _outGm[outLoadOffset], AlignUp(_outNumPerBatch, _dataNumPerBlock));

        int32_t idxValOffset = outHeadOffset - outLoadOffset;
        computeBatch(headID, outNum, outLoadOffset, idxValOffset);

        DataCopyPad(_outGm[outLoadOffset], _outLocal, {1, static_cast<uint32_t>(outNum * sizeof(DTYPE_OUT)), 0, 0, 0});
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    // traverse indices within headIDth HEAD
    __aicore__ inline void computeBatch(uint64_t headID, uint64_t outNum, 
        uint64_t outLoadOffset, int32_t idxValOffset)
    {
        _indicesMask = _maskBuf.Get<uint8_t>();
        uint64_t indicesHeadOffset = headID * _indicesNumPerHead;
        
        for (uint64_t n = 0; n < _indicesLoop; n++) {
            uint64_t indicesLoadOffset = indicesHeadOffset + n * _indicesNumPerBatch;
            uint64_t indicesLoadNum = calcBatchIndicesNum(n);
            uint64_t indicesLoadAlign = AlignUp(indicesLoadNum, _dataNumPerBlock);
            uint64_t cmpLenAlign256 = AlignUp(indicesLoadNum, MASK_ALIGN_SIZE / sizeof(DTYPE_INDICES));
            uint64_t mask64BitNum = ceilDiv(indicesLoadNum, 64);

            DataCopy(_indicesLocal, _indicesGm[indicesLoadOffset], indicesLoadAlign);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            DataCopy(_srcLocal, _srcGm[indicesLoadOffset], indicesLoadAlign);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            Adds(_indicesLocal, _indicesLocal, idxValOffset, indicesLoadAlign);
            Cast(_indicesLocal.ReinterpretCast<float>(), _indicesLocal, RoundMode::CAST_NONE, indicesLoadNum);
            CompareScalar(_indicesMask, _indicesLocal.ReinterpretCast<float>(), 0.0f, CMPMODE::GE, cmpLenAlign256);
            CompareScalar(_indicesMask[_maskLen], _indicesLocal.ReinterpretCast<float>(), outNum*1.0f, CMPMODE::LT, cmpLenAlign256);
            Cast(_indicesLocal, _indicesLocal.ReinterpretCast<float>(), RoundMode::CAST_RINT, indicesLoadNum);
            And(_indicesMask.ReinterpretCast<uint16_t>(), _indicesMask.ReinterpretCast<uint16_t>(),
                _indicesMask[_maskLen].ReinterpretCast<uint16_t>(), _maskLen);
            
            for (uint64_t m = 0; m < mask64BitNum; m++) {
                uint64_t mask = _indicesMask.ReinterpretCast<uint64_t>().GetValue(m);
                uint64_t maxPos = (m == mask64BitNum-1) ? indicesLoadNum - m * 64 : 64;

                for (int32_t p = ScalarGetSFFValue<1>(mask); p >= 0 && p < maxPos; p = ScalarGetSFFValue<1>(mask)) {
                    mask = sbitset0(mask, p);
                    uint64_t pos = p + m * 64;
                    DTYPE_INDICES idxVal = _indicesLocal.GetValue(pos);
                    _outLocal.SetValue(idxVal, _outLocal.GetValue(idxVal) + _srcLocal.GetValue(pos));
                }
            }
        }
    }

    __aicore__ inline uint64_t calcBatchOutNum(uint64_t k)
    {
        if (_outNumLeftover == 0) {
            return _outNumPerBatch;
        }
        return k == _outLoop - 1 ? _outNumLeftover : _outNumPerBatch;
    }

    __aicore__ inline uint64_t calcBatchIndicesNum(uint64_t n)
    {
        if (_indicesNumLeftover == 0) {
            return _indicesNumPerBatch;
        }
        return n == _indicesLoop - 1 ? _indicesNumLeftover : _indicesNumPerBatch;
    }

private:
    // common
    uint64_t _headIdOffset;
    uint64_t _bigCoreNum;
    uint64_t _outLoop;
    uint64_t _outNumPerBatch;
    uint64_t _outNumLeftover;
    uint64_t _maxOutNumPerBatch;
    uint64_t _outNumPartOffset;
    uint64_t _indicesLoop;
    uint64_t _indicesNumPerBatch;
    uint64_t _indicesNumLeftover;
    // in case large head
    uint64_t _headNumBigCore;
    uint64_t _headNumSmallCore;
    // in case head num less than core num 
    uint64_t _headPartId;
    uint64_t _coreNumPerHead;
    uint64_t _outNumPerCore;
    // mask operation
    TBuf<TPosition::VECCALC> _maskBuf;
    LocalTensor<uint8_t> _indicesMask;
    uint64_t _maskLen;
};

template<bool smallTail>
class ScatterAddWithTail final : public ScatterAddBaseKernel {
public:
    __aicore__ inline ScatterAddWithTail() = delete;

    __aicore__ inline ScatterAddWithTail(GM_ADDR src, GM_ADDR indices, 
        GM_ADDR out, ScatterAddTilingDataV3* tiling_data, TPipe* pipe) 
        : ScatterAddBaseKernel(src, indices, out, tiling_data, pipe)
    {
        if constexpr (smallTail) {
            initTilingSmallTail(tiling_data);
        } else {
            initTilingLargeTail(tiling_data);
        }
        initBuffer();
    }

    __aicore__ inline void Process()
    {
        for (uint64_t i = 0; i < _loop; i++) {
            if constexpr (smallTail) {
                computeWithSmallTail(i);
            } else {
                computeWithLargeTail(i);
            }
        }
    }

private:
    __aicore__ inline void initTilingSmallTail(ScatterAddTilingDataV3* tiling_data)
    {
        _srcTailBigCore = tiling_data->srcTailBigCore;
        _srcTailSmallCore = tiling_data->srcTailSmallCore;
        _bigCoreNum = tiling_data->bigCoreNum;

        _dbTimes = 1;
        _tailElemNum = AlignUp(_tailLen, _dataNumPerBlock);
        _tailNumPerBatch = min(_srcTailBigCore, _ubSize / sizeof(DTYPE_SRC) / (_tailElemNum + 1));
        _tailNumPerBatch = min(_tailNumPerBatch, MAX_COPY_PAD);
        _tailNumPerBatch = (_tailNumPerBatch == 0) ? 1 : _tailNumPerBatch;

        _indicesLoadLen = AlignUp(_tailNumPerBatch, _dataNumPerBlock);
        _srcLoadLen = _tailElemNum * _tailNumPerBatch;

        if (_blockIdx < _bigCoreNum) {
            _indicesBaseOffset = _blockIdx * _srcTailBigCore;
            _loop = ceilDiv(_srcTailBigCore, _tailNumPerBatch);
            _tailNumLeftover = _srcTailBigCore % _tailNumPerBatch;
        } else {
            _indicesBaseOffset = _bigCoreNum * _srcTailBigCore + (_blockIdx - _bigCoreNum) * _srcTailSmallCore;
            _loop = ceilDiv(_srcTailSmallCore, _tailNumPerBatch);
            _tailNumLeftover = _srcTailSmallCore % _tailNumPerBatch;
        }
    }

    __aicore__ inline void initTilingLargeTail(ScatterAddTilingDataV3* tiling_data)
    {
        _srcTailBigCore = tiling_data->srcTailBigCore;
        _srcTailSmallCore = tiling_data->srcTailSmallCore;
        _bigCoreNum = tiling_data->bigCoreNum;
        _tailLenThreshold = tiling_data->tailLenThreshold;

        _dbTimes = min(_ubSize / sizeof(DTYPE_SRC) / min(_tailLen, _tailLenThreshold), BUFFER_NUM_MAX);
        _dbTimes = (_dbTimes == 0) ? 1 : _dbTimes;

        uint64_t srcElemNumTmp = min(static_cast<uint64_t>(AlignUp(_tailLen, _dataNumPerBlock)), _tailLenThreshold) * _dbTimes;
        uint64_t availIndicesSize = _ubSize - srcElemNumTmp * sizeof(DTYPE_SRC);
        uint64_t srcTailNum = (_blockIdx < _bigCoreNum) ? _srcTailBigCore : _srcTailSmallCore;
       _indicesLoadLen = min(availIndicesSize / BLOCK_SIZE * BLOCK_SIZE / sizeof(DTYPE_INDICES), srcTailNum);
       _indicesLoadLen = (_indicesLoadLen == 0) ? 1 : _indicesLoadLen;

        _loop = ceilDiv(srcTailNum, _indicesLoadLen);
        _indicesNumPerBatch = _indicesLoadLen;
        _indicesNumLeftover = srcTailNum % _indicesLoadLen;

        _tailElemNum = (_ubSize - _indicesLoadLen * sizeof(DTYPE_INDICES)) / _dbTimes / BLOCK_SIZE * _dataNumPerBlock;
        _tailElemNum = min(_tailElemNum, static_cast<uint64_t>(AlignUp(_tailLen, _dataNumPerBlock)));
        _tailElemNum = (_tailElemNum == 0) ? 1 : _tailElemNum;
        _srcLoadLen = _tailElemNum * _dbTimes;

        _tailElemLoop = _tailLen / _tailElemNum;
        _tailElemNumLeftover = _tailLen % _tailElemNum;

        if (_blockIdx < _bigCoreNum) {
            _indicesBaseOffset = _blockIdx * _srcTailBigCore;
        } else {
            _indicesBaseOffset = _bigCoreNum * _srcTailBigCore + (_blockIdx - _bigCoreNum) * _srcTailSmallCore;
        }

        _countDB = 0;
        _eventID = EVENT_ID0;
    }

    __aicore__ inline void initBuffer()
    {
        _pipe->InitBuffer(_srcBuf, AlignUp(_srcLoadLen, _dataNumPerBlock) * sizeof(DTYPE_SRC));
        _pipe->InitBuffer(_indicesBuf, AlignUp(_indicesLoadLen, _dataNumPerBlock) * sizeof(DTYPE_INDICES));
    }

    __aicore__ inline void computeWithSmallTail(uint64_t i)
    {
        _srcLocal = _srcBuf.Get<DTYPE_SRC>();
        _indicesLocal = _indicesBuf.Get<DTYPE_INDICES>();
        
        uint64_t indicesOffset = _indicesBaseOffset + i * _tailNumPerBatch;
        uint64_t srcOffset = indicesOffset * _tailLen;
        uint16_t loadTailNum = calcTailLoadNum(i);

        pipe_barrier(PIPE_ALL);
        DataCopy(_indicesLocal, _indicesGm[indicesOffset], AlignUp(loadTailNum, _dataNumPerBlock));
        copyParamsIn = {loadTailNum, static_cast<uint32_t>(_tailLen * sizeof(DTYPE_SRC)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_OUT> queryCopyInPadParams{false, 0, 0, 0};
        DataCopyPad(_srcLocal, _srcGm[srcOffset], copyParamsIn, queryCopyInPadParams);

        SetAtomicAdd<DTYPE_SRC>();
        for (uint64_t p = 0; p < loadTailNum; p++) {
            DTYPE_INDICES idxVal = _indicesLocal.GetValue(p);
            uint64_t idxGlobal = indicesOffset + p;
            uint64_t headID = idxGlobal / _srcDimSize;
            uint64_t srcLocalOffset = p * AlignUp(_tailLen, _dataNumPerBlock);
            uint64_t outOffset = (idxVal + headID * _dimSize) * _tailLen;
            copyParamsOut = {1, static_cast<uint32_t>(_tailLen *sizeof(DTYPE_OUT)), 0, 0, 0};
            DataCopyPad(_outGm[outOffset], _srcLocal[srcLocalOffset], copyParamsOut);
        }
        SetAtomicNone();
    }
    
    __aicore__ inline void computeWithLargeTail(uint64_t i)
    {
        _srcLocal = _srcBuf.Get<DTYPE_SRC>();
        _indicesLocal = _indicesBuf.Get<DTYPE_INDICES>();

        uint64_t indicesOffset = _indicesBaseOffset + i * _indicesNumPerBatch;
        uint64_t indicesLoadNum = calcIndicesLoadNum(i);
        DataCopy(_indicesLocal, _indicesGm[indicesOffset], AlignUp(indicesLoadNum, _dataNumPerBlock));
        setFlagMET3_2();

        for (uint64_t p = 0; p < indicesLoadNum; p++) {
            DTYPE_INDICES idxVal = _indicesLocal.GetValue(p);
            uint64_t idxGlobal = indicesOffset + p;
            auto srcOffset = idxGlobal * _tailLen;
            
            SetAtomicAdd<DTYPE_SRC>();
            computeLargeTailAdd(idxGlobal, idxVal, srcOffset);
            SetAtomicNone();
        }
        waitFlagMET3_2();
    }

    __aicore__ inline void computeLargeTailAdd(uint64_t idxGlobal, uint64_t idxVal, uint64_t srcOffset)
    {
        uint64_t headID = idxGlobal / _srcDimSize;
        uint64_t outGlobalOffset = (idxVal + headID * _dimSize) * _tailLen;

        for (uint64_t k = 0; k < _tailElemLoop; k++) {
            uint64_t offset = k * _tailElemNum;
            uint64_t localOffset = getEventIdforDoublebuffer();
            WaitFlag<HardEvent::MTE3_MTE2>(_eventID);
            DataCopy(_srcLocal[localOffset], _srcGm[srcOffset + offset], _tailElemNum);
            SetFlag<HardEvent::MTE2_MTE3>(_eventID);
            WaitFlag<HardEvent::MTE2_MTE3>(_eventID);
            DataCopy(_outGm[outGlobalOffset + offset], _srcLocal[localOffset], _tailElemNum);
            SetFlag<HardEvent::MTE3_MTE2>(_eventID);
        }

        if (_tailElemNumLeftover != 0) {
            uint64_t offset = _tailElemLoop * _tailElemNum;
            uint64_t localOffset = getEventIdforDoublebuffer();
            WaitFlag<HardEvent::MTE3_MTE2>(_eventID);
            DataCopy(_srcLocal[localOffset], _srcGm[srcOffset + offset], AlignUp(_tailElemNumLeftover, _dataNumPerBlock));
            SetFlag<HardEvent::MTE2_MTE3>(_eventID);
            WaitFlag<HardEvent::MTE2_MTE3>(_eventID);
            copyParamsOut = {1, static_cast<uint32_t>(_tailElemNumLeftover * sizeof(DTYPE_SRC)), 0, 0, 0};
            DataCopyPad(_outGm[outGlobalOffset + offset], _srcLocal[localOffset], copyParamsOut);
            SetFlag<HardEvent::MTE3_MTE2>(_eventID);
        }
    }

    __aicore__ inline uint64_t getEventIdforDoublebuffer()
    {
        _eventID = _countDB % _dbTimes;
        uint64_t localOffset = _tailElemNum * _eventID;
        _countDB++;
        return localOffset;
    }

    __aicore__ inline void setFlagMET3_2()
    {
        for (uint64_t i = 0; i < _dbTimes; i++) {
            SetFlag<HardEvent::MTE3_MTE2>(i);
        }
    }

    __aicore__ inline void waitFlagMET3_2()
    {
        for (uint64_t i = 0; i < _dbTimes; i++) {
            WaitFlag<HardEvent::MTE3_MTE2>(i);
        }
    }

    __aicore__ inline uint64_t calcTailLoadNum(uint64_t i)
    {
        if (_tailNumLeftover == 0) {
            return _tailNumPerBatch;
        }
        return (i == _loop-1) ? _tailNumLeftover : _tailNumPerBatch;
    }

    __aicore__ inline uint64_t calcIndicesLoadNum(uint64_t i)
    {
        if (_indicesNumLeftover == 0) {
            return _indicesNumPerBatch;
        }
        return (i == _loop-1) ? _indicesNumLeftover : _indicesNumPerBatch;
    }

private:
    uint64_t _srcTailBigCore;
    uint64_t _srcTailSmallCore;
    uint64_t _bigCoreNum;
    uint64_t _tailLenThreshold;
    uint64_t _srcLoadLen;
    uint64_t _indicesLoadLen;
    uint64_t _indicesBaseOffset;
    uint64_t _tailElemNum;
    // in case small tail, processing tail by tail
    uint64_t _tailNumPerBatch;
    uint64_t _tailNumLeftover;
    // in case large tail, processing according to indices elem
    uint64_t _indicesNumPerBatch;
    uint64_t _indicesNumLeftover;
    uint64_t _tailElemLoop;
    uint64_t _tailElemNumLeftover;
    uint64_t _dbTimes;
    uint64_t _countDB;
    uint64_t _eventID;

    DataCopyExtParams copyParamsIn;
    DataCopyExtParams copyParamsOut;
};

#endif