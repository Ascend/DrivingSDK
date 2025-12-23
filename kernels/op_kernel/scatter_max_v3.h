/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 */
#ifndef _SCATTER_MAX_V3_H_
#define _SCATTER_MAX_V3_H_

#include "kernel_operator.h"

using namespace AscendC;

constexpr uint64_t MASK_ALIGN_SIZE = 256;


class KernelScatterMaxBase {
public:
    __aicore__ inline KernelScatterMaxBase() = delete;

    __aicore__ inline KernelScatterMaxBase(
        GM_ADDR src, GM_ADDR idx, GM_ADDR res, GM_ADDR argmax, ScatterMaxTilingDataV3 *tiling_data, TPipe* pipe)
         : _pipe(pipe)
    {
        uint64_t blockIdx = GetBlockIdx();
        uint64_t blockNum = GetBlockNum();

        ASSERT(blockNum != 0 && "block dim can not be zero!");

        _srcElemNum = tiling_data->srcElemNum;
        _idxElemNum = tiling_data->idxElemNum;
        _resElemNum = tiling_data->resElemNum;
        _tailElemNum = tiling_data->tailElemNum;
        _elemNumPerBlock = tiling_data->elemNumPerBlock;
        _tailElemNumAlign = AlignUp(_tailElemNum, _elemNumPerBlock);
        _tailSize = _tailElemNum * sizeof(DTYPE_SRC);
        _tailSizeAlign = _tailElemNumAlign * sizeof(DTYPE_SRC);

        _idxNumPerCore = tiling_data->idxNumPerCore;
        _idxBatchNum = tiling_data->idxBatchNum;
        _idxBatchNumAlign = AlignUp(_idxBatchNum, _elemNumPerBlock);
        _idxBaseOffset = _idxNumPerCore * blockIdx;
        _idxLoop = _idxNumPerCore ? ceilDiv(_idxNumPerCore, _idxBatchNum) : 0;

        _tailBatchNum = tiling_data->tailBatchNum;
        _srcBatchNum = tiling_data->srcBatchNum;
        _srcBatchNumAlign = AlignUp(_srcBatchNum, _elemNumPerBlock);

        _coreNumPerTail = tiling_data->coreNumPerTail;
        _leftSrcNumBigCore = tiling_data->leftSrcNumBigCore;
        _leftSrcBigCoreNum = tiling_data->leftSrcBigCoreNum;
        _leftSrcBatchNum = tiling_data->leftSrcBatchNum;
        _leftSrcBatchNumAlign = AlignUp(_leftSrcBatchNum, _elemNumPerBlock);

        if (_coreNumPerTail == 0) {
            _leftSrcIdxPos = _idxNumPerCore * blockNum;
        } else {
            _leftSrcIdxPos = _idxNumPerCore * blockNum + blockIdx / _coreNumPerTail;
        }
        
        uint64_t leftIdxNum = _idxElemNum % blockNum;
        if (blockIdx < leftIdxNum * _coreNumPerTail) {
            if (blockIdx % _coreNumPerTail < _leftSrcBigCoreNum) {
                _leftSrcBaseOffset = _leftSrcIdxPos * _tailElemNum + (blockIdx % _coreNumPerTail) * _leftSrcNumBigCore;
                _leftSrcNumCurCore = _leftSrcNumBigCore;
            } else {
                _leftSrcBaseOffset = _leftSrcIdxPos * _tailElemNum + _leftSrcBigCoreNum * _leftSrcNumBigCore
                         + (blockIdx % _coreNumPerTail - _leftSrcBigCoreNum) * (_leftSrcNumBigCore - 1);
                _leftSrcNumCurCore = _leftSrcNumBigCore - 1;
            }
        } else {
            _leftSrcBaseOffset = 0;
            _leftSrcNumCurCore = 0;
        }
        _leftSrcLoop = (_leftSrcBatchNum == 0) ? 0 : ceilDiv(_leftSrcNumCurCore, _leftSrcBatchNum);

        _srcGM.SetGlobalBuffer((__gm__ DTYPE_SRC*)src, _srcElemNum);
        _resGM.SetGlobalBuffer((__gm__ DTYPE_RES*)res, _resElemNum);
        _idxGM.SetGlobalBuffer((__gm__ DTYPE_INDEX*)idx, _idxElemNum);
        _argmaxGM.SetGlobalBuffer((__gm__ DTYPE_ARGMAX*)argmax, _resElemNum);
    }

protected:
    template <typename T1, typename T2>
    __aicore__ inline T1 ceilDiv(T1 a, T2 b) {
        return b == 0 ? 0 : (a + b - 1) / b;
    };

protected:
    TPipe* _pipe;
    TBuf<TPosition::VECCALC> _srcBuf;
    TBuf<TPosition::VECCALC> _idxBuf;

    GlobalTensor<DTYPE_SRC> _srcGM;
    GlobalTensor<DTYPE_INDEX> _idxGM;
    GlobalTensor<DTYPE_RES> _resGM;
    GlobalTensor<DTYPE_ARGMAX> _argmaxGM;

    LocalTensor<DTYPE_SRC> _srcLocal;
    LocalTensor<DTYPE_INDEX> _idxLocal;

    uint64_t _srcElemNum;
    uint64_t _idxElemNum;
    uint64_t _resElemNum;
    uint64_t _tailElemNum;
    uint64_t _tailElemNumAlign;
    uint64_t _tailSize;
    uint64_t _tailSizeAlign;
    uint64_t _elemNumPerBlock;

    uint64_t _idxNumPerCore;
    uint64_t _idxBatchNum;
    uint64_t _idxBatchNumAlign;
    uint64_t _idxBaseOffset;
    uint64_t _idxLoop;

    uint64_t _tailBatchNum;
    uint64_t _srcBatchNum;
    uint64_t _srcBatchNumAlign;
    uint64_t _srcLoop;

    uint64_t _coreNumPerTail;
    uint64_t _leftSrcNumBigCore;
    uint64_t _leftSrcBigCoreNum;
    uint64_t _leftSrcNumCurCore;
    uint64_t _leftSrcBatchNum;
    uint64_t _leftSrcBatchNumAlign;
    uint64_t _leftSrcIdxPos;
    uint64_t _leftSrcBaseOffset;
    uint64_t _leftSrcLoop;
};

template<bool smallTail>
class KernelScatterMaxV3 : public KernelScatterMaxBase {
public:
    __aicore__ inline KernelScatterMaxV3() = delete;

    __aicore__ inline KernelScatterMaxV3(
        GM_ADDR src, GM_ADDR idx, GM_ADDR res, GM_ADDR argmax, 
        ScatterMaxTilingDataV3 *tiling_data, TPipe* pipe)
        : KernelScatterMaxBase(src, idx, res, argmax, tiling_data, pipe)
    {
        if constexpr (smallTail) {
            _srcLoop = 0;
        } else {
            _srcLoop = ceilDiv(_tailElemNum, _srcBatchNum);
        }
    }

public:
    __aicore__ inline void Process()
    {
        initBatchProcessBuffer();
        for (uint64_t i = 0; i < _idxLoop; i++) {
            batchProcess(i);
        }

        _pipe->Reset();
        initLeftSrcBuffer();
        for (uint64_t i = 0; i < _leftSrcLoop; i++) {
            processLeftSrc(i);
        }
    }

private:
    __aicore__ inline void initBatchProcessBuffer()
    {
        _pipe->InitBuffer(_idxBuf, _idxBatchNumAlign * sizeof(DTYPE_INDEX));
        if constexpr (smallTail) {
            _pipe->InitBuffer(_srcBuf, _tailBatchNum * _tailSizeAlign);
        } else {
            _pipe->InitBuffer(_srcBuf, _srcBatchNumAlign * sizeof(DTYPE_SRC));
        }
    }

    __aicore__ inline void initLeftSrcBuffer()
    {
        _pipe->InitBuffer(_idxBuf, _elemNumPerBlock * sizeof(DTYPE_INDEX));
        _pipe->InitBuffer(_srcBuf, _leftSrcBatchNumAlign * sizeof(DTYPE_SRC));
    }

    __aicore__ inline void batchProcess(uint64_t i)
    {
        uint64_t idxOffset = _idxBaseOffset + i * _idxBatchNum;
        uint64_t idxLoadNum = min(_idxBatchNum, _idxNumPerCore - i * _idxBatchNum);
        uint64_t idxLoadNumAlgin = AlignUp(idxLoadNum, _elemNumPerBlock);
        uint64_t tailLoop = ceilDiv(idxLoadNum, _tailBatchNum);

        _idxLocal = _idxBuf.Get<DTYPE_INDEX>();
        DataCopy(_idxLocal, _idxGM[idxOffset], idxLoadNumAlgin);

        if constexpr (smallTail) {
            for (uint64_t k = 0; k < tailLoop; k++) {
                tailWisebatchProcess(k, idxOffset, idxLoadNum);
            }
        } else {
            for (uint64_t k = 0; k < idxLoadNum; k++) {
                elemWiseBatchProcess(k, idxOffset);
            }
        }
    }

    __aicore__ inline void tailWisebatchProcess(uint64_t k, uint64_t idxOffset, uint64_t idxLoadNum)
    {
        uint64_t tailOffset = idxOffset + k * _tailBatchNum;
        uint64_t tailLoadNum = min(_tailBatchNum, idxLoadNum - k * _tailBatchNum);
        DataCopyExtParams copyParams = {static_cast<uint16_t>(tailLoadNum), static_cast<uint32_t>(_tailSize), 0, 0, 0};

        _srcLocal = _srcBuf.Get<DTYPE_SRC>();
        DataCopyPad(_srcLocal, _srcGM[tailOffset * _tailElemNum], copyParams, {0, 0, 0, 0});
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);

        SetAtomicMax<DTYPE_RES>();
        for (uint64_t n = 0; n < tailLoadNum; n++) {
            DTYPE_INDEX idxVal = _idxLocal.GetValue(k * _tailBatchNum + n);
            DataCopyPad(_resGM[idxVal * _tailElemNum], _srcLocal[n * _tailElemNumAlign], {1, static_cast<uint32_t>(_tailSize), 0, 0, 0});
        }
        SetAtomicNone();
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void elemWiseBatchProcess(uint64_t k, uint64_t idxOffset)
    {
        DTYPE_INDEX idxVal = _idxLocal.GetValue(idxOffset + k);

        for (uint64_t n = 0; n < _srcLoop; n++) {
            uint64_t srcOffset = (idxOffset + k) * _tailElemNum + n * _srcBatchNum;
            uint64_t srcLoadNum = min(_srcBatchNum, _tailElemNum - n * _srcBatchNum);
            uint64_t srcLoadNumAlign = AlignUp(srcLoadNum, _elemNumPerBlock);
            
            _srcLocal = _srcBuf.Get<DTYPE_SRC>();
            DataCopy(_srcLocal, _srcGM[srcOffset], srcLoadNumAlign);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            
            SetAtomicMax<DTYPE_RES>();
            DataCopyPad(_resGM[idxVal * _tailElemNum], _srcLocal, {1, static_cast<uint32_t>(srcLoadNum * sizeof(DTYPE_RES)), 0, 0, 0});
            SetAtomicNone();
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void processLeftSrc(uint64_t i)
    {
        uint64_t srcOffset = _leftSrcBaseOffset + i * _leftSrcBatchNum;
        uint64_t srcLoadNum = min(_leftSrcBatchNum, _leftSrcNumCurCore - i * _leftSrcBatchNum);
        uint64_t srcLoadNumAlign = AlignUp(srcLoadNum, _elemNumPerBlock);

        _idxLocal = _idxBuf.Get<DTYPE_INDEX>();
        _srcLocal = _srcBuf.Get<DTYPE_SRC>();

        DataCopy(_idxLocal, _idxGM[_leftSrcIdxPos], _elemNumPerBlock);
        DataCopy(_srcLocal, _srcGM[srcOffset], srcLoadNumAlign);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        
        DTYPE_INDEX idxVal = _idxLocal.GetValue(0);
        uint64_t resOffset = idxVal * _tailElemNum + srcOffset % _tailElemNum;

        SetAtomicMax<DTYPE_RES>();
        DataCopyPad(_resGM[resOffset], _srcLocal, {1, static_cast<uint32_t>(srcLoadNum * sizeof(DTYPE_RES)), 0, 0, 0});
        SetAtomicNone();
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
};

template<bool smallTail>
class KernelScatterMaxArgmaxV3 : public KernelScatterMaxBase {
public:
    __aicore__ inline KernelScatterMaxArgmaxV3() = delete;

    __aicore__ inline KernelScatterMaxArgmaxV3(
        GM_ADDR src, GM_ADDR idx, GM_ADDR res, GM_ADDR argmax, 
        ScatterMaxTilingDataV3 *tiling_data, TPipe* pipe)
        : KernelScatterMaxBase(src, idx, res, argmax, tiling_data, pipe)
    {
        if constexpr (smallTail) {
            _srcLoop = 0;
        } else {
            _srcLoop = ceilDiv(_tailElemNum, _srcBatchNum);
        }
    }

public:
    __aicore__ inline void Process()
    {
        initBatchProcessBuffer();
        for (uint64_t i = 0; i < _idxLoop; i++) {
            batchProcess(i);
        }

        _pipe->Reset();
        initLeftSrcBuffer();
        for (uint64_t i = 0; i < _leftSrcLoop; i++) {
            processLeftSrc(i);
        }
    }

private:
    __aicore__ inline void initBatchProcessBuffer()
    {
        uint64_t maskBitNum = AscendCUtils::GetBitSize(sizeof(uint8_t));

        if constexpr (smallTail) {
            _pipe->InitBuffer(_srcBuf, _tailBatchNum * _tailSizeAlign);
            _pipe->InitBuffer(_resBuf, _tailSizeAlign);
            _pipe->InitBuffer(_argmaxBuf, _tailSizeAlign);
            _srcMaskNum = AlignUp(_tailSize, MASK_ALIGN_SIZE) / sizeof(DTYPE_SRC);
        } else {
            _pipe->InitBuffer(_srcBuf, _srcBatchNumAlign * sizeof(DTYPE_SRC));
            _pipe->InitBuffer(_resBuf, _srcBatchNumAlign * sizeof(DTYPE_RES));
            _pipe->InitBuffer(_argmaxBuf, _srcBatchNumAlign * sizeof(DTYPE_ARGMAX));
            _srcMaskNum = AlignUp(_srcBatchNum * sizeof(DTYPE_SRC), MASK_ALIGN_SIZE) / sizeof(DTYPE_SRC);
        }

        uint64_t maskBufSize = ceilDiv(_srcMaskNum, maskBitNum) * sizeof(uint8_t);
        _pipe->InitBuffer(_maskBuf, maskBufSize);
        _pipe->InitBuffer(_idxBuf, _idxBatchNumAlign * sizeof(DTYPE_INDEX));
    }

    __aicore__ inline void initLeftSrcBuffer()
    {
        _pipe->InitBuffer(_idxBuf, _elemNumPerBlock * sizeof(DTYPE_INDEX));
        _pipe->InitBuffer(_srcBuf, _leftSrcBatchNumAlign * sizeof(DTYPE_SRC));
        _pipe->InitBuffer(_resBuf, _leftSrcBatchNumAlign * sizeof(DTYPE_RES));
        _pipe->InitBuffer(_argmaxBuf, _leftSrcBatchNumAlign * sizeof(DTYPE_ARGMAX));

        _srcMaskNum = AlignUp(_leftSrcBatchNumAlign * sizeof(DTYPE_SRC), MASK_ALIGN_SIZE) / sizeof(DTYPE_SRC);
        uint64_t maskBitNum = AscendCUtils::GetBitSize(sizeof(uint8_t));
        uint64_t maskBufSize = ceilDiv(_srcMaskNum, maskBitNum) * sizeof(uint8_t);
        _pipe->InitBuffer(_maskBuf, maskBufSize);
    }

    __aicore__ inline void batchProcess(uint64_t i)
    {
        uint64_t idxOffset = _idxBaseOffset + i * _idxBatchNum;
        uint64_t idxLoadNum = min(_idxBatchNum, _idxNumPerCore - i * _idxBatchNum);
        uint64_t idxLoadNumAlgin = AlignUp(idxLoadNum, _elemNumPerBlock);
        uint64_t tailLoop = ceilDiv(idxLoadNum, _tailBatchNum);

        _idxLocal = _idxBuf.Get<DTYPE_INDEX>();
        DataCopy(_idxLocal, _idxGM[idxOffset], idxLoadNumAlgin);

        if constexpr (smallTail) {
            for (uint64_t k = 0; k < tailLoop; k++) {
                tailWisebatchProcess(k, idxOffset, idxLoadNum);
            }
        } else {
            for (uint64_t k = 0; k < idxLoadNum; k++) {
                elemWiseBatchProcess(k, idxOffset);
            }
        }
    }

    __aicore__ inline void tailWisebatchProcess(uint64_t k, uint64_t idxOffset, uint64_t idxLoadNum)
    {
        uint64_t tailOffset = idxOffset + k * _tailBatchNum;
        uint64_t tailLoadNum = min(_tailBatchNum, idxLoadNum - k * _tailBatchNum);
        DataCopyExtParams copyParams = {static_cast<uint16_t>(tailLoadNum), static_cast<uint32_t>(_tailSize), 0, 0, 0};

        _srcLocal = _srcBuf.Get<DTYPE_SRC>();
        _resLocal = _resBuf.Get<DTYPE_RES>();
        _argmaxLocal = _argmaxBuf.Get<DTYPE_ARGMAX>();
        auto _argmaxFloatLocal = _argmaxLocal.ReinterpretCast<float>();
        _maskLocal = _maskBuf.Get<uint8_t>();

        DataCopyPad(_srcLocal, _srcGM[tailOffset * _tailElemNum], copyParams, {0, 0, 0, 0});

        SetAtomicMax<DTYPE_ARGMAX>();
        for (uint64_t n = 0; n < tailLoadNum; n++) {
            DTYPE_INDEX idxVal = _idxLocal.GetValue(k * _tailBatchNum + n);
            uint64_t resOffset = idxVal * _tailElemNum;
            int64_t srcGlobalPos = tailOffset + n;
            
            DataCopy(_resLocal, _resGM[resOffset], _tailElemNumAlign);
            DataCopy(_argmaxLocal, _argmaxGM[resOffset], _tailElemNumAlign);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            Compare(_maskLocal, _srcLocal[n * _tailElemNumAlign], _resLocal, CMPMODE::NE, _srcMaskNum);
            Cast(_argmaxFloatLocal, _argmaxLocal, RoundMode::CAST_NONE, _tailElemNumAlign);
            Select(_argmaxFloatLocal, _maskLocal, _argmaxFloatLocal, static_cast<float>(srcGlobalPos), SELMODE::VSEL_TENSOR_SCALAR_MODE, _tailElemNum);
            Cast(_argmaxLocal, _argmaxFloatLocal, RoundMode::CAST_RINT, _tailElemNumAlign);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

            DataCopyPad(_argmaxGM[resOffset], _argmaxLocal, {1, static_cast<uint32_t>(_tailSize), 0, 0, 0});
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        SetAtomicNone();
    }


    __aicore__ inline void elemWiseBatchProcess(uint64_t k, uint64_t idxOffset)
    {
        DTYPE_INDEX idxVal = _idxLocal.GetValue(idxOffset + k);

        _srcLocal = _srcBuf.Get<DTYPE_SRC>();
        _resLocal = _resBuf.Get<DTYPE_RES>();
        _argmaxLocal = _argmaxBuf.Get<DTYPE_ARGMAX>();
        auto _argmaxFloatLocal = _argmaxLocal.ReinterpretCast<float>();
        _maskLocal = _maskBuf.Get<uint8_t>();

        for (uint64_t n = 0; n < _srcLoop; n++) {
            int64_t idxPos = idxOffset + k;
            uint64_t resOffset = idxVal * _tailElemNum + n * _srcBatchNum;
            uint64_t srcOffset = idxPos * _tailElemNum + n * _srcBatchNum;
            uint64_t srcLoadNum = min(_srcBatchNum, _tailElemNum - n * _srcBatchNum);
            uint64_t srcLoadNumAlign = AlignUp(srcLoadNum, _elemNumPerBlock);

            DataCopy(_srcLocal, _srcGM[srcOffset], srcLoadNumAlign);
            DataCopy(_resLocal, _resGM[resOffset], srcLoadNumAlign);
            DataCopy(_argmaxLocal, _argmaxGM[resOffset], srcLoadNumAlign);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            Compare(_maskLocal, _srcLocal, _resLocal, CMPMODE::NE, _srcMaskNum);
            Cast(_argmaxFloatLocal, _argmaxLocal, RoundMode::CAST_NONE, srcLoadNumAlign);
            Select(_argmaxFloatLocal, _maskLocal, _argmaxFloatLocal, static_cast<float>(idxPos), SELMODE::VSEL_TENSOR_SCALAR_MODE, _tailElemNum);
            Cast(_argmaxLocal, _argmaxFloatLocal, RoundMode::CAST_RINT, srcLoadNumAlign);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            
            SetAtomicMax<DTYPE_ARGMAX>();
            DataCopyPad(_argmaxGM[resOffset], _argmaxLocal, {1, static_cast<uint32_t>(srcLoadNum * sizeof(DTYPE_ARGMAX)), 0, 0, 0});
            SetAtomicNone();
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void processLeftSrc(uint64_t i)
    {
        uint64_t srcOffset = _leftSrcBaseOffset + i * _leftSrcBatchNum;
        uint64_t srcLoadNum = min(_leftSrcBatchNum, _leftSrcNumCurCore - i * _leftSrcBatchNum);
        uint64_t srcLoadNumAlign = AlignUp(srcLoadNum, _elemNumPerBlock);
        int64_t idxPos = _leftSrcIdxPos;

        _idxLocal = _idxBuf.Get<DTYPE_INDEX>();
        _srcLocal = _srcBuf.Get<DTYPE_SRC>();
        _resLocal = _resBuf.Get<DTYPE_RES>();
        _argmaxLocal = _argmaxBuf.Get<DTYPE_ARGMAX>();
        auto _argmaxFloatLocal = _argmaxLocal.ReinterpretCast<float>();
        _maskLocal = _maskBuf.Get<uint8_t>();

        DataCopy(_idxLocal, _idxGM[_leftSrcIdxPos], _elemNumPerBlock);
        DataCopy(_srcLocal, _srcGM[srcOffset], srcLoadNumAlign);
        
        DTYPE_INDEX idxVal = _idxLocal.GetValue(0);
        uint64_t resOffset = idxVal * _tailElemNum + srcOffset % _tailElemNum;

        DataCopy(_resLocal, _resGM[resOffset], srcLoadNumAlign);
        DataCopy(_argmaxLocal, _argmaxGM[resOffset], srcLoadNumAlign);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        Compare(_maskLocal, _srcLocal, _resLocal, CMPMODE::NE, _srcMaskNum);
        Cast(_argmaxFloatLocal, _argmaxLocal, RoundMode::CAST_NONE, srcLoadNumAlign);
        Select(_argmaxFloatLocal, _maskLocal, _argmaxFloatLocal, static_cast<float>(idxPos), SELMODE::VSEL_TENSOR_SCALAR_MODE, srcLoadNum);
        Cast(_argmaxLocal, _argmaxFloatLocal, RoundMode::CAST_RINT, srcLoadNumAlign);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

        SetAtomicMax<DTYPE_ARGMAX>();
        DataCopyPad(_argmaxGM[resOffset], _argmaxLocal, {1, static_cast<uint32_t>(srcLoadNum * sizeof(DTYPE_ARGMAX)), 0, 0, 0});
        SetAtomicNone();
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

private:
    TBuf<TPosition::VECCALC> _resBuf;
    TBuf<TPosition::VECCALC> _argmaxBuf;
    TBuf<TPosition::VECCALC> _maskBuf;

    LocalTensor<DTYPE_RES> _resLocal;
    LocalTensor<DTYPE_ARGMAX> _argmaxLocal;
    LocalTensor<uint8_t> _maskLocal;

    uint64_t _srcMaskNum;
};

#endif
