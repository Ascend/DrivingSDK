/*
* Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
*/
#include "kernel_operator.h"
using namespace AscendC;


namespace AscendC {
template<typename Ta, typename Tb>
__aicore__ inline Ta min(const Ta a, const Tb b)
{
    if (a > b) {
        return b;
    }
    return a;
}

template<typename Ta, typename Tb>
__aicore__ inline Ta max(const Ta a, const Tb b)
{
    if (a < b) {
        return b;
    }
    return a;
}

template<typename T>
class KernelUnique {
public:
    __aicore__ inline KernelUnique(TPipe& pipe) : pipe(pipe) {}
    // Each block process diffent part of data. This function returns the element-wise first index of data by blockIdx.
    __aicore__ inline size_t GetGlobalOffset(const uint32_t blockIdx);
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR uniqueCnt, GM_ADDR workspace,
        const uint32_t totalLength, const uint32_t totalTileNum, const uint32_t shortBlockTileNum,
        const uint16_t tailLength, const uint8_t blockNum, const uint8_t shortBlockNum);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(const int32_t progress);
    __aicore__ inline void Elem32Sort(const int32_t progress);
    __aicore__ inline void TileSort(const int32_t progress);
    template<typename T1>
    __aicore__ inline static void DataCopyGM2GM(const GlobalTensor<T1>& dst, const GlobalTensor<T1>& src,
        const LocalTensor<T1>& tmpLocal, const int elemLength, const int bufByteLength);
    using GMSSrcList = GlobalTensor<float> (&)[4];
    struct GMSParams {
        int (&GMSLengths)[4];
        uint8_t& queNum;
        LocalTensor<float> (&&buffLocal)[5];
    };
    __aicore__ inline static void MrgSortGM(GlobalTensor<float>&& dstGlobal, GMSSrcList& srcList, GMSParams& params);
    __aicore__ inline void BlockSortV2();
    __aicore__ inline void GlobalSortV2();
    __aicore__ inline static void ConsecutiveUnique(const LocalTensor<float>& dstVal,
        const LocalTensor<float>& srcLocal, const LocalTensor<float>& shiftedLocal,
        const LocalTensor<uint32_t>& bitMask16, const uint16_t elemLength, uint64_t& tileUniqueCnt);
    __aicore__ inline void TileUnique(const int32_t progress);
    __aicore__ inline void CopyOut();

private:
    static constexpr int32_t TILE_LENGTH = 8192;
    // INF to fill the tail blank, so that tail is automatically removed by Compare in Unique.
    static constexpr float FLOAT_INF = 3e+99;
    // Indicates the factor converting float to data structure used by Sort32&MrgSort.
    static constexpr int16_t SORT_DATATYPE_SIZE = sizeof(float) + sizeof(uint32_t);          // 8
    static constexpr int16_t SORT_DATATYPE_SIZE_FACTOR = SORT_DATATYPE_SIZE / sizeof(float); // 2
    static constexpr int32_t TILE_LEN_BYTE = TILE_LENGTH * SORT_DATATYPE_SIZE;               // 8192 * 8 = 65536
    static constexpr int32_t TILE_LEN_ELEM = TILE_LENGTH * SORT_DATATYPE_SIZE_FACTOR;        // 8192 * 2 = 16384
    static constexpr uint16_t VALID_QUE[5] = {
        0, 0, 0b11, 0b111, 0b1111}; // Converts queue number to validBit of MrgSort.

    TPipe& pipe;
    TQue<TPosition::VECIN, 1> calcBuf[3];

    GlobalTensor<T> srcGlobal;
    GlobalTensor<uint32_t> srcGlobalAsUint;
    GlobalTensor<T> dstGlobal1;
    GlobalTensor<int32_t> dstGlobal1As32;
    GlobalTensor<int32_t> uniqueCntGlobal;

    GlobalTensor<float> sortedBlock1;
    GlobalTensor<int32_t> sortedBlock1AsInt;
    GlobalTensor<float> sortedBlock2;
    GlobalTensor<int32_t> sortedBlock2AsInt;
    GlobalTensor<float> sortedGlobal1;
    GlobalTensor<float> sortedGlobal2;

    GlobalTensor<int32_t> IBSyncGlobal;
    GlobalTensor<uint32_t> blockUniqueCntGlobal;

    uint16_t syncWorkspaceSize;
    uint8_t eventID {0};
    uint64_t accUniqueCnt {0};
    float lastTileUniqueVal;

    uint32_t totalLength;
    uint32_t alignedTotalLength;
    uint32_t tileNum;
    uint32_t shortBlockTileNum;
    uint16_t tailLength;
    uint8_t blockNum;
    uint8_t shortBlockNum;

    size_t globalOffset; // Offset of data for current block.
    size_t blockLength;  // Length of current block.
    bool hasInfFlag {false};
};

// Each block process diffent part of data. This function returns the element-wise first index of data by blockIdx.
template<typename T>
__aicore__ inline size_t KernelUnique<T>::GetGlobalOffset(const uint32_t blockIdx)
{
    // (shortBlockTileNum + 1) indicates longBlockTileNum.
    const size_t offset =
        (this->shortBlockTileNum * min(this->shortBlockNum, blockIdx) +
            (this->shortBlockTileNum + 1) * (this->shortBlockNum >= blockIdx ? 0 : blockIdx - this->shortBlockNum)) *
        TILE_LENGTH;
    return offset;
}

template<typename T>
__aicore__ inline void KernelUnique<T>::Init(GM_ADDR input, GM_ADDR output, GM_ADDR uniqueCnt, GM_ADDR workspace,
    const uint32_t totalLength, const uint32_t totalTileNum, const uint32_t shortBlockTileNum,
    const uint16_t tailLength, const uint8_t blockNum, const uint8_t shortBlockNum)
{
    this->totalLength = totalLength;
    this->alignedTotalLength = totalTileNum * TILE_LENGTH;
    this->shortBlockTileNum = shortBlockTileNum;
    this->tailLength = tailLength;
    this->blockNum = blockNum;
    this->shortBlockNum = shortBlockNum;

    const bool isShortBlock = this->shortBlockNum > GetBlockIdx();
    // (shortBlockTileNum + 1) indicates longBlockTileNum.
    this->tileNum = isShortBlock ? shortBlockTileNum : shortBlockTileNum + 1;
    this->blockLength = this->tileNum * TILE_LENGTH;
    this->globalOffset = GetGlobalOffset(GetBlockIdx());

    srcGlobal.SetGlobalBuffer((__gm__ T*)input + globalOffset, this->blockLength);
    srcGlobalAsUint.SetGlobalBuffer((__gm__ uint32_t*)input + globalOffset * sizeof(T) / sizeof(uint32_t),
        this->blockLength * sizeof(T) / sizeof(uint32_t));
    dstGlobal1.SetGlobalBuffer((__gm__ T*)output, this->alignedTotalLength);
    dstGlobal1As32.SetGlobalBuffer((__gm__ int32_t*)output, this->alignedTotalLength * sizeof(T) / sizeof(int32_t));
    uniqueCntGlobal.SetGlobalBuffer((__gm__ int32_t*)uniqueCnt, 1);

    // sortedBlock is offsetted, and could only see the data that this block should process.
    sortedBlock1.SetGlobalBuffer((__gm__ float*)workspace + globalOffset * SORT_DATATYPE_SIZE_FACTOR,
        this->blockLength * SORT_DATATYPE_SIZE_FACTOR);
    sortedBlock1AsInt.SetGlobalBuffer((__gm__ int32_t*)workspace + globalOffset * SORT_DATATYPE_SIZE_FACTOR,
        this->blockLength * SORT_DATATYPE_SIZE_FACTOR);
    sortedBlock2.SetGlobalBuffer((__gm__ float*)workspace + alignedTotalLength * SORT_DATATYPE_SIZE_FACTOR +
                                    globalOffset * SORT_DATATYPE_SIZE_FACTOR,
        this->blockLength * SORT_DATATYPE_SIZE_FACTOR);
    sortedBlock2AsInt.SetGlobalBuffer((__gm__ int32_t*)workspace + alignedTotalLength * SORT_DATATYPE_SIZE_FACTOR +
                                        globalOffset * SORT_DATATYPE_SIZE_FACTOR,
        this->blockLength * SORT_DATATYPE_SIZE_FACTOR);
    // sortedGlobal could see all data in the workspace.
    sortedGlobal1.SetGlobalBuffer((__gm__ float*)workspace, alignedTotalLength * SORT_DATATYPE_SIZE_FACTOR);
    sortedGlobal2.SetGlobalBuffer((__gm__ float*)workspace + alignedTotalLength * SORT_DATATYPE_SIZE_FACTOR,
        alignedTotalLength * SORT_DATATYPE_SIZE_FACTOR);

    // Buff size for syncronizing according to document of IBWait&IBSet.
    this->syncWorkspaceSize = (blockNum * 32 + 1) * 8;
    IBSyncGlobal.SetGlobalBuffer(
        (__gm__ int32_t*)workspace + alignedTotalLength * SORT_DATATYPE_SIZE_FACTOR * 2, syncWorkspaceSize);
    blockUniqueCntGlobal.SetGlobalBuffer((__gm__ uint32_t*)workspace + alignedTotalLength * 4 + syncWorkspaceSize,
        (blockNum + 7) / 8 * 8); // Length aligned up to 32B.

    pipe.InitBuffer(calcBuf[0], 1, TILE_LEN_BYTE);
    pipe.InitBuffer(calcBuf[1], 1, TILE_LEN_BYTE);
    pipe.InitBuffer(calcBuf[2], 1, TILE_LEN_BYTE);
}

template<typename T>
__aicore__ inline void KernelUnique<T>::Process()
{
    LocalTensor<int32_t> IBSyncLocal;
    // Initialize sync buff.
    if (GetBlockIdx() == 0) {
        IBSyncLocal = calcBuf[0].AllocTensor<int32_t>();
        Duplicate(IBSyncLocal, 0, syncWorkspaceSize);
        PipeBarrier<PIPE_ALL>();
        DataCopy(IBSyncGlobal, IBSyncLocal, syncWorkspaceSize);
        PipeBarrier<PIPE_ALL>();
        calcBuf[0].FreeTensor(IBSyncLocal);
    } // Initialize sync buff.

    // Sort within each tile.
    for (int32_t tileIdx = 0; tileIdx < this->tileNum; tileIdx++) {
        CopyIn(tileIdx);
        Elem32Sort(tileIdx);
        TileSort(tileIdx);
    }

    if (GetBlockNum() > 1) {
        if (this->tileNum > 1) {
            BlockSortV2(); // Sort within each block.
        }

        GlobalSortV2(); // Sort globally.

        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        PipeBarrier<PIPE_ALL>();
    }

    // Check if an inf value exists. If do, inf will be append to the result in TileUnique().
    if ((IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value || IsSameType<T, float>::value) &&
        GetBlockIdx() == blockNum - 1) {
        PipeBarrier<PIPE_ALL>();
        const uint32_t totalLength32BAligned = (totalLength * sizeof(T) + 31) / 32 * 32 / sizeof(float);
        if (sortedGlobal1.GetValue((totalLength32BAligned - 1) * 2) == -FLOAT_INF) {
            hasInfFlag = true;
        }
    }

    // Do unique in each block based on tiles.
    for (int32_t tileIdx = 0; tileIdx < this->tileNum; tileIdx++) {
        TileUnique(tileIdx);
    }

    if (this->blockNum > 1) {
        // Each block waits for its former block to upload blockUniqueCnt.
        IBSyncLocal = calcBuf[0].AllocTensor<int32_t>();
        if (GetBlockIdx() != 0) {
            PipeBarrier<PIPE_ALL>();
            IBWait(IBSyncGlobal, IBSyncLocal, (int32_t)GetBlockIdx() - 1, eventID);
            PipeBarrier<PIPE_ALL>();
        }
        PipeBarrier<PIPE_ALL>();
        IBSet(IBSyncGlobal, IBSyncLocal, (int32_t)GetBlockIdx(), eventID);
        PipeBarrier<PIPE_ALL>();
        calcBuf[0].FreeTensor(IBSyncLocal);
    }

    // Gather result from every block.
    CopyOut();
}

template<typename T>
__aicore__ inline void KernelUnique<T>::CopyIn(const int32_t progress)
{
    LocalTensor<T> srcLocal = calcBuf[0].AllocTensor<T>();
    LocalTensor<float> sortedLocal2 = calcBuf[2].AllocTensor<float>();

    // To process tail, fill the whole tile with INF, then cover it with tail.
    int32_t castLen; // Valid length of the last block.
    if ((progress != tileNum - 1) || (GetBlockIdx() != blockNum - 1) || tailLength == 0) {
        // Must determine during compilation, otherwise we get a compilation error.
        if constexpr (!IsSameType<T, float>::value) {
            DataCopy(srcLocal, srcGlobal[progress * TILE_LENGTH], TILE_LENGTH);
        } else {
            DataCopy(sortedLocal2, srcGlobal[progress * TILE_LENGTH], TILE_LENGTH);
        }
        castLen = TILE_LENGTH;
    } else {
        // Process tail.
        LocalTensor<uint32_t> srcAsUint = srcLocal.template ReinterpretCast<uint32_t>();
        Duplicate(sortedLocal2, FLOAT_INF, TILE_LENGTH);
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(sortedLocal2, srcGlobal[progress * TILE_LENGTH],
                {1, static_cast<uint16_t>(sizeof(T) * tailLength), 0, 0}, {false, 0, 0, 0});
        } else if constexpr (sizeof(T) >= sizeof(float)) {
            DataCopyPad(srcAsUint, srcGlobalAsUint[progress * TILE_LENGTH * sizeof(T) / sizeof(uint32_t)],
                {1, static_cast<uint16_t>(sizeof(T) * tailLength), 0, 0}, {false, 0, 0, 0});
        } else {
            DataCopyPad(srcLocal, srcGlobal[progress * TILE_LENGTH],
                {1, static_cast<uint16_t>(sizeof(T) * tailLength), 0, 0}, {false, 0, 0, 0});
        }
        castLen = tailLength;
    }
    PipeBarrier<PIPE_ALL>();
    if constexpr (!IsSameType<T, float>::value) {
        if constexpr (sizeof(T) >= sizeof(float)) {
            Cast(sortedLocal2, srcLocal, RoundMode::CAST_ROUND, castLen);
        } else {
            Cast(sortedLocal2, srcLocal, RoundMode::CAST_NONE, castLen);
        }
        PipeBarrier<PIPE_V>();
    }
    Muls(sortedLocal2, sortedLocal2, (float)-1, TILE_LENGTH);
    calcBuf[0].EnQue(srcLocal);
    calcBuf[2].EnQue(sortedLocal2);
}

template<typename T>
__aicore__ inline void KernelUnique<T>::Elem32Sort(const int32_t progress)
{
    LocalTensor<T> srcLocal = calcBuf[0].DeQue<T>();
    LocalTensor<float> sortedLocal1 = calcBuf[1].AllocTensor<float>();
    LocalTensor<float> sortedLocal2 = calcBuf[2].DeQue<float>();
    LocalTensor<int32_t> arithLocal = srcLocal.template ReinterpretCast<int32_t>()[TILE_LENGTH];

    int32_t baseOffset = progress * TILE_LENGTH + this->globalOffset; // calc tileOffset
    Duplicate(arithLocal, baseOffset, TILE_LENGTH);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint32_t> uidArray = arithLocal.template ReinterpretCast<uint32_t>();
    // Max repeatTime of Sort32 is 255, which is exceeded because TILE_LENGTH is 8192.
    constexpr uint8_t sort32BatchSize = 32;
    constexpr uint8_t sort32RepeatLimit = 255;
    int instrRepeatTime = 0;
    int restLen = TILE_LENGTH;
    while (restLen) {
        int repTime = min(restLen / sort32BatchSize, sort32RepeatLimit);
        Sort32<float>(sortedLocal1[sort32BatchSize * sort32RepeatLimit * SORT_DATATYPE_SIZE_FACTOR * instrRepeatTime],
            sortedLocal2[sort32BatchSize * sort32RepeatLimit * instrRepeatTime],
            uidArray[sort32BatchSize * sort32RepeatLimit * instrRepeatTime], repTime);
        restLen -= repTime * sort32BatchSize;
        instrRepeatTime++;
    }
    PipeBarrier<PIPE_ALL>();
    calcBuf[0].FreeTensor(srcLocal);
    calcBuf[1].EnQue(sortedLocal1);
    calcBuf[2].EnQue(sortedLocal2);
}

template<typename T>
__aicore__ inline void KernelUnique<T>::TileSort(const int32_t progress)
{
    LocalTensor<float> sortedLocal1 = calcBuf[1].DeQue<float>();
    LocalTensor<float> sortedLocal2 = calcBuf[2].DeQue<float>();
    LocalTensor<float> sortedQue[2] = {sortedLocal1, sortedLocal2};
    uint16_t currentQueLength = 32; // Initial queue length is 32 because data is from Sort32.
    uint16_t currentQueNum = TILE_LENGTH / currentQueLength;
    bool switchFlag = false;
    // Multiple MrgSort until we have one generally sorted tile.
    while (currentQueLength < TILE_LENGTH) {
        const uint16_t elementLengths[4] = {currentQueLength, currentQueLength, currentQueLength, currentQueLength};
        const uint16_t fullMrgSortTime = currentQueNum / 4;
        if (fullMrgSortTime > 0) {
            MrgSort4Info params = {elementLengths, false, 0b1111, fullMrgSortTime};
            MrgSort<float>(sortedQue[!switchFlag],
                {sortedQue[switchFlag][0], sortedQue[switchFlag][currentQueLength * 1 * 2],
                    sortedQue[switchFlag][currentQueLength * 2 * 2], sortedQue[switchFlag][currentQueLength * 3 * 2]},
                params);
            PipeBarrier<PIPE_ALL>();
            switchFlag = !switchFlag;
        }
        currentQueNum = fullMrgSortTime;
        currentQueLength *= 4;
    }
    DataCopy(sortedBlock1[progress * TILE_LEN_ELEM], sortedQue[switchFlag], TILE_LEN_ELEM);
    PipeBarrier<PIPE_ALL>();
    calcBuf[1].FreeTensor(sortedLocal1);
    calcBuf[2].FreeTensor(sortedLocal2);
}

template<typename T>
template<typename T1>
__aicore__ inline void KernelUnique<T>::DataCopyGM2GM(const GlobalTensor<T1>& dst, const GlobalTensor<T1>& src,
    const LocalTensor<T1>& tmpLocal, const int elemLength, const int bufByteLength)
{
    // Max byte size of DataCopyPad in one repeat is 65535.
    int bufElemLength = min(bufByteLength, 65535) / sizeof(T1);
    int restLen = elemLength;
    while (restLen > 0) {
        int copyLen = min(restLen, bufElemLength);
        DataCopyPad(tmpLocal, src[elemLength - restLen], {1, static_cast<uint16_t>(sizeof(T1) * copyLen), 0, 0},
            {false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(dst[elemLength - restLen], tmpLocal, {1, static_cast<uint16_t>(sizeof(T1) * copyLen), 0, 0});
        PipeBarrier<PIPE_ALL>();
        restLen -= copyLen;
    }
}

template<typename T>
__aicore__ inline void KernelUnique<T>::MrgSortGM(
    GlobalTensor<float>&& dstGlobal, GMSSrcList& srcList, GMSParams& params)
{
    int restLen[4] {params.GMSLengths[0], params.GMSLengths[1], params.GMSLengths[2], params.GMSLengths[3]};
    int currentHead[4] {};
    int totalMrgLen {};
    uint8_t queNum = params.queNum;
    // limited by MrgSort api constraint and mrgLocal size, we set different buffer length due to diffent queNum.
    // mrgLocal contains 8192 elems, and MrgSort limits max 4095 elems per queue.
    constexpr int BUFFER_LEN[5] {0, 0, 4095, 2730, 2048};
    uint16_t sortedLen[4];
    uint16_t mrgLen[4] {};
    while (queNum > 1) {
        int currentBufferLen = BUFFER_LEN[queNum];
        for (int i = 0; i < queNum; i++) {
            mrgLen[i] = min(restLen[i], currentBufferLen);
        }
        // CopyIn
        for (int i = 0; i < queNum; i++) {
            DataCopyPad(params.buffLocal[i], srcList[i][currentHead[i] * SORT_DATATYPE_SIZE_FACTOR],
                {1, static_cast<uint16_t>(sizeof(float) * mrgLen[i] * SORT_DATATYPE_SIZE_FACTOR), 0, 0},
                {false, 0, 0, 0});
        }
        PipeBarrier<PIPE_ALL>();
        // MrgSort
        MrgSort4Info localParams {mrgLen, true, VALID_QUE[queNum], 1};
        MrgSort<float>(params.buffLocal[4],
            {params.buffLocal[0], params.buffLocal[1], params.buffLocal[2], params.buffLocal[3]}, localParams);
        PipeBarrier<PIPE_ALL>();
        GetMrgSortResult(sortedLen[0], sortedLen[1], sortedLen[2], sortedLen[3]);
        const uint16_t localMrgLen = sortedLen[0] + sortedLen[1] + sortedLen[2] + sortedLen[3];
        // CopyOut
        DataCopyPad(dstGlobal[totalMrgLen * SORT_DATATYPE_SIZE_FACTOR], params.buffLocal[4],
            {1, static_cast<uint16_t>(sizeof(float) * localMrgLen * SORT_DATATYPE_SIZE_FACTOR), 0, 0});
        PipeBarrier<PIPE_ALL>();
        // renew currentHead, restLen
        totalMrgLen += localMrgLen;
        for (int i = 0; i < queNum; i++) {
            restLen[i] -= sortedLen[i];
            currentHead[i] += sortedLen[i];
        }
        // Switch empty to tail
        for (int i = 0; i < queNum; i++) {
            if (restLen[i] == 0) {
                for (int j = i; j < 3; j++) {
                    restLen[j] = restLen[j + 1];
                    currentHead[j] = currentHead[j + 1];
                    srcList[j] = srcList[j + 1];
                }
                restLen[3] = 0;
                queNum--;
                break; // because ifExhaustedSuspension == true, there is 0 or 1 empty que.
            }
        }
    }
    // Process tail
    for (int i = 0; i < params.queNum; i++) {
        if (restLen[i] > 0) {
            DataCopyGM2GM(dstGlobal[totalMrgLen * SORT_DATATYPE_SIZE_FACTOR],
                srcList[i][currentHead[i] * SORT_DATATYPE_SIZE_FACTOR], params.buffLocal[4],
                restLen[i] * SORT_DATATYPE_SIZE_FACTOR, TILE_LEN_BYTE);
            break;
        }
    }
};

template<typename T>
__aicore__ inline void KernelUnique<T>::BlockSortV2()
{
    LocalTensor<float> sortedLocal1 = calcBuf[0].AllocTensor<float>();
    LocalTensor<float> sortedLocal2 = calcBuf[1].AllocTensor<float>();
    LocalTensor<float> mrgLocal = calcBuf[2].AllocTensor<float>();
    GlobalTensor<float> sortedBlock[2] = {sortedBlock1, sortedBlock2};

    // Each time merge 4 queues into 1 queue.
    constexpr uint8_t PREFIX_QUE_NUM = 4;
    bool switchFlag = false;
    GlobalTensor<float> srcGlobal[4];
    LocalTensor<float> buffLocal[5];
    int lengths[4];
    for (int bindTile = 1; bindTile < tileNum; bindTile *= PREFIX_QUE_NUM) {
        for (int tileIdx = 0; tileIdx < tileNum; tileIdx += bindTile * PREFIX_QUE_NUM) {
            int mrgTileNum = min(tileNum - tileIdx, bindTile * PREFIX_QUE_NUM);
            uint8_t queNum = (mrgTileNum + bindTile - 1) / bindTile;
            uint8_t lastQueTileNum = mrgTileNum % bindTile;
            if (lastQueTileNum == 0) {
                lastQueTileNum = bindTile;
            }
            // Init GMSSrcList, GMSParams
            for (int i = 0; i < queNum; i++) {
                srcGlobal[i] = sortedBlock[switchFlag][TILE_LEN_ELEM * (tileIdx + bindTile * i)];
            }
            for (int i = 0; i < queNum - 1; i++) {
                lengths[i] = TILE_LENGTH * bindTile;
            }
            lengths[queNum - 1] = TILE_LENGTH * lastQueTileNum;
            GMSSrcList srcList {srcGlobal};
            GMSParams params {lengths, queNum,
                {sortedLocal1, sortedLocal1[TILE_LENGTH], sortedLocal2, sortedLocal2[TILE_LENGTH], mrgLocal}};
            MrgSortGM(sortedBlock[!switchFlag][TILE_LEN_ELEM * tileIdx], srcList, params);
        }
        switchFlag = !switchFlag;
    }
    if (switchFlag) {
        DataCopyGM2GM(sortedBlock1, sortedBlock2, sortedLocal1, blockLength * SORT_DATATYPE_SIZE_FACTOR, TILE_LEN_BYTE);
    }
    calcBuf[0].FreeTensor(sortedLocal1);
    calcBuf[1].FreeTensor(sortedLocal2);
    calcBuf[2].FreeTensor(mrgLocal);
}

template<typename T>
__aicore__ inline void KernelUnique<T>::GlobalSortV2()
{
    LocalTensor<float> sortedLocal1 = calcBuf[0].AllocTensor<float>();
    LocalTensor<float> sortedLocal2 = calcBuf[1].AllocTensor<float>();
    LocalTensor<float> mrgLocal = calcBuf[2].AllocTensor<float>();
    LocalTensor<int32_t> IBSyncLocal = sortedLocal2.ReinterpretCast<int32_t>();
    GlobalTensor<float> sortedGlobal[2] = {sortedGlobal1, sortedGlobal2};

    // Each time merge up to 4 queues into 1 queue.
    constexpr uint8_t PREFIX_QUE_NUM = 4;
    bool switchFlag = false;
    GlobalTensor<float> srcGlobal[4];
    int lengths[4];
    for (int bindBlock = 1; bindBlock < blockNum; bindBlock *= PREFIX_QUE_NUM, eventID++) {
        for (int blockIdx = 0; blockIdx < blockNum; blockIdx += bindBlock * PREFIX_QUE_NUM) {
            if ((GetBlockIdx() == blockIdx + bindBlock) || (GetBlockIdx() == blockIdx + bindBlock * 2) ||
                (GetBlockIdx() == blockIdx + bindBlock * 3)) {
                PipeBarrier<PIPE_ALL>();
                IBSet(IBSyncGlobal, IBSyncLocal, (int32_t)GetBlockIdx(), eventID);
                PipeBarrier<PIPE_ALL>();
            } else if (GetBlockIdx() == blockIdx) {
                int mrgBlockNum = min(blockNum - blockIdx, bindBlock * PREFIX_QUE_NUM);
                uint8_t queNum = (mrgBlockNum + bindBlock - 1) / bindBlock;
                for (int i = 1; i < queNum; i++) {
                    PipeBarrier<PIPE_ALL>();
                    IBWait(IBSyncGlobal, IBSyncLocal, (int32_t)blockIdx + (bindBlock * i), eventID);
                    PipeBarrier<PIPE_ALL>();
                }
                // 判断最后一个队列包含了多少个block的数据.
                uint8_t lastQueBlockNum = mrgBlockNum % bindBlock;
                if (lastQueBlockNum == 0) {
                    lastQueBlockNum = bindBlock;
                }
                // Init GMSSrcList, GMSParams
                for (int i = 0; i < queNum; i++) {
                    srcGlobal[i] =
                        sortedGlobal[switchFlag][GetGlobalOffset(blockIdx + bindBlock * i) * SORT_DATATYPE_SIZE_FACTOR];
                }
                for (int i = 0; i < queNum - 1; i++) {
                    lengths[i] =
                        GetGlobalOffset(blockIdx + (bindBlock * (i + 1))) - GetGlobalOffset(blockIdx + (bindBlock * i));
                }
                lengths[queNum - 1] = GetGlobalOffset(blockIdx + (bindBlock * (queNum - 1)) + lastQueBlockNum) -
                                    GetGlobalOffset(blockIdx + (bindBlock * (queNum - 1)));
                GMSSrcList srcList {srcGlobal};
                GMSParams params {lengths, queNum,
                    {sortedLocal1, sortedLocal1[TILE_LENGTH], sortedLocal2, sortedLocal2[TILE_LENGTH], mrgLocal}};
                MrgSortGM(
                    sortedGlobal[!switchFlag][GetGlobalOffset(blockIdx) * SORT_DATATYPE_SIZE_FACTOR], srcList, params);
            }
        }
        switchFlag = !switchFlag;
    }
    // Switch valid workspace pointer.
    if (switchFlag) {
        GlobalTensor<float> tmpGlobal = sortedGlobal1;
        sortedGlobal1 = sortedGlobal2;
        sortedGlobal2 = tmpGlobal;

        GlobalTensor<float> tmpGlobal1 = sortedBlock1;
        sortedBlock1 = sortedBlock2;
        sortedBlock2 = tmpGlobal1;

        GlobalTensor<int32_t> tmpGlobal2 = sortedBlock1AsInt;
        sortedBlock1AsInt = sortedBlock2AsInt;
        sortedBlock2AsInt = tmpGlobal2;
    }
    calcBuf[0].FreeTensor(sortedLocal1);
    calcBuf[1].FreeTensor(sortedLocal2);
    calcBuf[2].FreeTensor(mrgLocal);
}

template<typename T>
__aicore__ inline void KernelUnique<T>::ConsecutiveUnique(const LocalTensor<float>& dstVal,
    const LocalTensor<float>& srcLocal, const LocalTensor<float>& shiftedLocal, const LocalTensor<uint32_t>& bitMask32,
    const uint16_t elemLength, uint64_t& tileUniqueCnt)
{
    LocalTensor<uint16_t> bitMask16 = bitMask32.ReinterpretCast<uint16_t>();
    uint64_t rsvdCnt = 0;
    // Seperate Val and Idx.
    GatherMask(dstVal, srcLocal, 1, false, 0, {1, static_cast<uint16_t>((elemLength * 2 + 63) / 64), 8, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();

    // Gen bitMask to calc shifted array.
    Duplicate(bitMask16, (uint16_t)0b1111111111111111, elemLength / 16);
    PipeBarrier<PIPE_V>();
    bitMask16.SetValue(0, 0b1111111111111110);

    // Calc shifted array.
    GatherMask(shiftedLocal, dstVal, bitMask32, true, elemLength, {1, 1, 8, 8}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    // Set the last val as INF in order to avoid dropping the last unique val.
    shiftedLocal.SetValue(elemLength - 1, -FLOAT_INF);

    // Generate bitMask which represents unique numbers.
    Compare(bitMask16, dstVal, shiftedLocal, CMPMODE::NE, (elemLength + 63) / 64 * 64);
    PipeBarrier<PIPE_V>();

    // Gather unique numbers and their idx.
    GatherMask(dstVal, dstVal, bitMask32, true, elemLength, {1, 1, 8, 8}, tileUniqueCnt);
    PipeBarrier<PIPE_V>();
}

template<typename T>
__aicore__ inline void KernelUnique<T>::TileUnique(const int32_t progress)
{
    LocalTensor<uint32_t> bitMask32 = calcBuf[0].AllocTensor<uint32_t>();
    LocalTensor<float> shiftedLocal = bitMask32[TILE_LENGTH].ReinterpretCast<float>();
    LocalTensor<float> sortedLocal1 = calcBuf[1].AllocTensor<float>();
    LocalTensor<float> sortedLocal2 = calcBuf[2].AllocTensor<float>();
    LocalTensor<uint32_t> uniqueCntLocal = shiftedLocal.ReinterpretCast<uint32_t>();
    uint64_t tileUniqueCnt;
    uint64_t tmpRsvdCnt;

    DataCopy(sortedLocal1, sortedBlock1[progress * TILE_LEN_ELEM], TILE_LEN_ELEM);
    PipeBarrier<PIPE_ALL>();

    ConsecutiveUnique(sortedLocal2, sortedLocal1, shiftedLocal, bitMask32, TILE_LENGTH, tileUniqueCnt);
    // If has inf, append.
    if ((progress == tileNum - 1) && hasInfFlag) {
        sortedLocal2.SetValue(tileUniqueCnt, -FLOAT_INF);
        tileUniqueCnt++;
    }

    if (tileUniqueCnt != 0) {
        accUniqueCnt += tileUniqueCnt;
        if (progress != 0 && lastTileUniqueVal == sortedLocal2.GetValue(0)) {
            accUniqueCnt--;
        }
        DataCopyPad(sortedBlock1[accUniqueCnt - tileUniqueCnt], sortedLocal2,
            {1, static_cast<uint16_t>(sizeof(float) * tileUniqueCnt), 0, 0});
        PipeBarrier<PIPE_ALL>();
        lastTileUniqueVal = sortedLocal2.GetValue(tileUniqueCnt - 1);
    }

    // upload uniqueCnt.
    if (progress == tileNum - 1) {
        uniqueCntLocal.SetValue(0, accUniqueCnt);
        DataCopyPad(blockUniqueCntGlobal[GetBlockIdx()], uniqueCntLocal,
            {1, static_cast<uint16_t>(sizeof(uint32_t) * 1), 0, 0});
        PipeBarrier<PIPE_ALL>();
    }
    calcBuf[0].FreeTensor(shiftedLocal);
    calcBuf[1].FreeTensor(sortedLocal1);
    calcBuf[2].FreeTensor(sortedLocal2);
}

template<typename T>
__aicore__ inline void KernelUnique<T>::CopyOut()
{
    LocalTensor<T> copyLocal0 = calcBuf[0].AllocTensor<T>();
    LocalTensor<float> copyLocal1 = calcBuf[1].AllocTensor<float>();
    LocalTensor<int32_t> IBSyncLocal = copyLocal1.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> copyLocal2 = calcBuf[2].AllocTensor<int32_t>();

    uint64_t lastAccUniqueCnt = 0;
    // Get every blockUniqueCnt before current block. Calc accumulate uniqueCnt.
    for (int i = 0; i < GetBlockIdx(); i++) {
        uint64_t lastUniqueCnt = blockUniqueCntGlobal.GetValue(i);
        lastAccUniqueCnt += lastUniqueCnt;
        // If the first val of (i+1)th block equals to the last val of (i)th block, then they should be placed in
        // the same position, accUniqueCnt--.
        if (sortedGlobal1[GetGlobalOffset(i + 1) * SORT_DATATYPE_SIZE_FACTOR].GetValue(0) ==
            sortedGlobal1[GetGlobalOffset(i) * SORT_DATATYPE_SIZE_FACTOR].GetValue(lastUniqueCnt - 1)) {
            lastAccUniqueCnt--;
        }
    }
    uint64_t thisUniqueCnt = blockUniqueCntGlobal.GetValue(GetBlockIdx());

    uint64_t restLen = thisUniqueCnt;
    // max(Ta a, Tb b) function does not support compilation period calc.
    constexpr uint64_t bottleneckTypeSize = sizeof(T) > sizeof(float) ? sizeof(T) : sizeof(float);
    LocalTensor<int32_t> copyVal32 = copyLocal0.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> uniqueVal32 = copyLocal1.ReinterpretCast<int32_t>();
    // Copy unique values (and counts) from Workspace to dst.
    while (restLen > 0) {
        // DataCopyPad could copy up to 65535B in one cycle. And one tile may contain up to 65536B. So we should
        // process multiple cycles.
        uint64_t copyLen = min(restLen, TILE_LEN_BYTE / bottleneckTypeSize);
        copyLen = min(copyLen, 65535 / bottleneckTypeSize);
        if constexpr (!IsSameType<T, float>::value) {
            DataCopyPad(copyLocal1, sortedBlock1[thisUniqueCnt - restLen],
                {1, static_cast<uint16_t>(sizeof(float) * copyLen), 0, 0}, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            Muls(copyLocal1, copyLocal1, (float)-1, copyLen);
            PipeBarrier<PIPE_V>();
            Cast(copyLocal0, copyLocal1, RoundMode::CAST_RINT, copyLen);
            PipeBarrier<PIPE_ALL>();
        } else {
            DataCopyPad(copyLocal0, sortedBlock1[thisUniqueCnt - restLen],
                {1, static_cast<uint16_t>(sizeof(float) * copyLen), 0, 0}, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            Muls(copyLocal0, copyLocal0, (float)-1, copyLen);
            PipeBarrier<PIPE_V>();
        }
        // DataCopyPad does not support int64_t. Copy them as uint32_t.
        if constexpr (sizeof(T) > 4) {
            DataCopyPad(dstGlobal1As32[(lastAccUniqueCnt + thisUniqueCnt - restLen) * sizeof(T) / sizeof(uint32_t)],
                copyVal32, {1, static_cast<uint16_t>(sizeof(T) * copyLen), 0, 0});
        } else {
            DataCopyPad(dstGlobal1[lastAccUniqueCnt + thisUniqueCnt - restLen], copyLocal0,
                {1, static_cast<uint16_t>(sizeof(T) * copyLen), 0, 0});
        }
        PipeBarrier<PIPE_ALL>();
        restLen -= copyLen;
    }
    // Return unique count.
    if (GetBlockIdx() == blockNum - 1) {
        uniqueVal32.SetValue(0, lastAccUniqueCnt + thisUniqueCnt);
        DataCopyPad(uniqueCntGlobal, uniqueVal32, {1, static_cast<uint16_t>(sizeof(uint32_t) * 1), 0, 0});
        PipeBarrier<PIPE_ALL>();
    }
    calcBuf[0].FreeTensor(copyLocal0);
    calcBuf[1].FreeTensor(copyLocal1);
}
} // namespace AscendC
