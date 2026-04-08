#include <cstdint>

using namespace AscendC;

constexpr float POSITIVE_INF = 1e24;
constexpr uint32_t VERTICES_ALIGNED = 32;
constexpr uint32_t TASK_SIZE_ALIGNED = 8;
constexpr uint32_t VERTICES_COUNT = 24;
constexpr uint32_t FLOAT_BYTE_SIZE = 4;


__aicore__ inline void ParseXYZWHDRBox(LocalTensor<float>& xLocal, LocalTensor<float>& yLocal, LocalTensor<float>& wLocal,
    LocalTensor<float>& hLocal, LocalTensor<float>& rLocal, LocalTensor<float>& boxInput, LocalTensor<uint32_t>& patternLocal_, uint32_t boxCount)
{
    bool reduceMode = false;
    uint32_t mask = 0;
    uint64_t rsvdCnt = 0;
    uint16_t repeatTimes = Ceil(boxCount * 8, 64);

    GatherMask(xLocal, boxInput, patternLocal_[0], reduceMode, mask, { 1, repeatTimes, 8, 0 }, rsvdCnt);
    GatherMask(yLocal, boxInput, patternLocal_[64], reduceMode, mask, { 1, repeatTimes, 8, 0 }, rsvdCnt);
    GatherMask(wLocal, boxInput, patternLocal_[128], reduceMode, mask, { 1, repeatTimes, 8, 0 }, rsvdCnt);
    GatherMask(hLocal, boxInput, patternLocal_[192], reduceMode, mask, { 1, repeatTimes, 8, 0 }, rsvdCnt);
    GatherMask(rLocal, boxInput, patternLocal_[256], reduceMode, mask, { 1, repeatTimes, 8, 0 }, rsvdCnt);
}

__aicore__ inline void CompareLess(LocalTensor<float>& maskLocal, LocalTensor<float>& src0Local, LocalTensor<float>& src1Local,
    LocalTensor<float>& tmpLocal, const uint32_t calCount)
{
    Sub(maskLocal, src1Local, src0Local, calCount);
    Sign(tmpLocal, maskLocal, calCount);
    Relu(maskLocal, tmpLocal, calCount);
}

__aicore__ inline void CompareLessEqual(LocalTensor<float>& maskLocal, LocalTensor<float>& src0Local, LocalTensor<float>& src1Local,
    LocalTensor<float>& tmpLocal, const uint32_t calCount)
{
    Sub(maskLocal, src1Local, src0Local, calCount);
    Sign(tmpLocal, maskLocal, calCount);
    Adds(maskLocal, tmpLocal, 1.0f, calCount);
    Mins(maskLocal, maskLocal, 1.0f, calCount);
}

__aicore__ inline void SortVertices(LocalTensor<int32_t>& SortedVerticesLocal, LocalTensor<float>& xVerticesLocal, LocalTensor<float>& yVerticesLocal,
    LocalTensor<float>& maskLocal, LocalTensor<int32_t>& numValidLocal, LocalTensor<int32_t>& sortIdx,
    LocalTensor<float>& tmpLocal, const uint32_t boxesCount, const bool reuseBuf)
{
    uint32_t calCount = boxesCount * VERTICES_ALIGNED;
    uint32_t maskCalCount = Ceil(calCount, 16);
    uint32_t boxesCountAligned = Ceil(boxesCount, TASK_SIZE_ALIGNED) * TASK_SIZE_ALIGNED;
    uint32_t mask = 0;
    uint64_t rsvdCnt = calCount;
    uint16_t repeatTimes = Ceil(calCount * 2, static_cast<uint32_t>(64));
    uint8_t src1Pattern = 2;
    uint32_t eventMTE3V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));

    LocalTensor<int32_t> firstSortedIdxLocal = tmpLocal.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> gatherLocal = tmpLocal[boxesCountAligned].ReinterpretCast<int32_t>();

    LocalTensor<float> xCeneterValLocal = tmpLocal[boxesCountAligned * 2];
    LocalTensor<float> yCeneterValLocal = tmpLocal[boxesCountAligned * 3];
    LocalTensor<float> numValidFloatLocal = tmpLocal[boxesCountAligned * 4];

    LocalTensor<float> tmpXVerticesLocal = tmpLocal[boxesCountAligned * 5];
    LocalTensor<float> tmpYVerticesLocal = tmpLocal[boxesCountAligned * 5 + VERTICES_ALIGNED * boxesCountAligned];
    LocalTensor<float> tmpXVerticesLocal1;
    LocalTensor<float> tmpYVerticesLocal1;
    LocalTensor<uint8_t> tmpMaskLocal1;
    LocalTensor<uint8_t> tmpMaskLocal2;
    LocalTensor<uint8_t> tmpMaskLocal3;
    if (reuseBuf) {
        tmpXVerticesLocal1 = xVerticesLocal;
        tmpYVerticesLocal1 = yVerticesLocal;
        tmpMaskLocal1 = tmpLocal[boxesCountAligned * 5 + VERTICES_ALIGNED * boxesCountAligned * 2].ReinterpretCast<uint8_t>();
        tmpMaskLocal2 = tmpLocal[boxesCountAligned * 6 + VERTICES_ALIGNED * boxesCountAligned * 2].ReinterpretCast<uint8_t>();
        tmpMaskLocal3 = tmpLocal[boxesCountAligned * 7 + VERTICES_ALIGNED * boxesCountAligned * 2].ReinterpretCast<uint8_t>();
    } else {
        tmpXVerticesLocal1 = tmpLocal[boxesCountAligned * 5 + VERTICES_ALIGNED * boxesCountAligned * 2];
        tmpYVerticesLocal1 = tmpLocal[boxesCountAligned * 5 + VERTICES_ALIGNED * boxesCountAligned * 3];
        tmpMaskLocal1 = tmpLocal[boxesCountAligned * 5 + VERTICES_ALIGNED * boxesCountAligned * 4].ReinterpretCast<uint8_t>();
        tmpMaskLocal2 = tmpLocal[boxesCountAligned * 6 + VERTICES_ALIGNED * boxesCountAligned * 4].ReinterpretCast<uint8_t>();
        tmpMaskLocal3 = tmpLocal[boxesCountAligned * 7 + VERTICES_ALIGNED * boxesCountAligned * 4].ReinterpretCast<uint8_t>();
    }
    LocalTensor<int32_t> idxLocal1 = tmpXVerticesLocal.ReinterpretCast<int32_t>();

    uint32_t broadCastSrcShape1[2] = {boxesCount, 1};
    uint32_t broadCastDstShape1[2] = {boxesCount, VERTICES_ALIGNED};

    CreateVecIndex(gatherLocal, 0, boxesCountAligned);
    Muls(gatherLocal, gatherLocal, static_cast<int32_t>(VERTICES_ALIGNED * 4), boxesCountAligned);
    Cast(numValidFloatLocal, numValidLocal, RoundMode::CAST_NONE, boxesCountAligned);
    Duplicate(maskLocal[VERTICES_COUNT], 0.0f, 8, boxesCount, 1, 4);    // 24 - 32 padding pos fill 0

    CompareScalar(tmpMaskLocal1, maskLocal, 1.0f, CMPMODE::EQ, boxesCountAligned * VERTICES_ALIGNED);
    Select(tmpXVerticesLocal1, tmpMaskLocal1, xVerticesLocal, 0.0f,
        SELMODE::VSEL_TENSOR_SCALAR_MODE, calCount);
    Select(tmpYVerticesLocal1, tmpMaskLocal1, yVerticesLocal, 0.0f,
        SELMODE::VSEL_TENSOR_SCALAR_MODE, calCount);

    WholeReduceSum<float>(xCeneterValLocal, tmpXVerticesLocal1, VERTICES_COUNT, boxesCount, 1, 1, 4);
    WholeReduceSum<float>(yCeneterValLocal, tmpYVerticesLocal1, VERTICES_COUNT, boxesCount, 1, 1, 4);
    Div(xCeneterValLocal, xCeneterValLocal, numValidFloatLocal, boxesCountAligned);
    Div(yCeneterValLocal, yCeneterValLocal, numValidFloatLocal, boxesCountAligned);
    BroadCast<float, 2, 1, false>(tmpXVerticesLocal, xCeneterValLocal, broadCastDstShape1, broadCastSrcShape1);
    BroadCast<float, 2, 1, false>(tmpYVerticesLocal, yCeneterValLocal, broadCastDstShape1, broadCastSrcShape1);
    Sub(tmpXVerticesLocal1, tmpXVerticesLocal1, tmpXVerticesLocal, calCount);
    Sub(tmpYVerticesLocal1, tmpYVerticesLocal1, tmpYVerticesLocal, calCount);
    
    Muls(maskLocal, maskLocal, -POSITIVE_INF, calCount);
    Adds(maskLocal, maskLocal, static_cast<float>(POSITIVE_INF), calCount);

    // ComputeRadian, store in tmpXVerticesLocal
    Div(tmpXVerticesLocal, tmpYVerticesLocal1, tmpXVerticesLocal1, calCount);
    Atan(tmpYVerticesLocal, tmpXVerticesLocal, calCount);
    Select(tmpYVerticesLocal, tmpMaskLocal1, tmpYVerticesLocal, 0.0f,
        SELMODE::VSEL_TENSOR_SCALAR_MODE, calCount);
    
    CompareScalar(tmpMaskLocal1, tmpXVerticesLocal1, 0.0f, CMPMODE::LT, boxesCountAligned * VERTICES_ALIGNED);
    CompareScalar(tmpMaskLocal2, tmpYVerticesLocal1, 0.0f, CMPMODE::LE, boxesCountAligned * VERTICES_ALIGNED);
    Not(tmpMaskLocal3.ReinterpretCast<uint16_t>(), tmpMaskLocal2.ReinterpretCast<uint16_t>(), maskCalCount);
    And(tmpMaskLocal3.ReinterpretCast<uint16_t>(), tmpMaskLocal1.ReinterpretCast<uint16_t>(), tmpMaskLocal3.ReinterpretCast<uint16_t>(), maskCalCount);
    And(tmpMaskLocal1.ReinterpretCast<uint16_t>(), tmpMaskLocal1.ReinterpretCast<uint16_t>(), tmpMaskLocal2.ReinterpretCast<uint16_t>(), maskCalCount);
    
    Adds(tmpXVerticesLocal1, tmpYVerticesLocal, -PI, calCount);
    Adds(tmpYVerticesLocal1, tmpYVerticesLocal, PI, calCount);

    Select(tmpXVerticesLocal, tmpMaskLocal1, tmpXVerticesLocal1, tmpYVerticesLocal,
        SELMODE::VSEL_TENSOR_TENSOR_MODE, calCount);
    Select(tmpXVerticesLocal, tmpMaskLocal3, tmpYVerticesLocal1, tmpXVerticesLocal,
        SELMODE::VSEL_TENSOR_TENSOR_MODE, calCount);
    
    // vertices_radian[~mask] = INF
    Add(tmpXVerticesLocal, maskLocal, tmpXVerticesLocal, calCount);
    // argsort
    Duplicate(tmpXVerticesLocal[VERTICES_COUNT], POSITIVE_INF, 8, boxesCount, 1, 4);    // 24 - 32 padding pos fill inf
    Muls(tmpXVerticesLocal, tmpXVerticesLocal, static_cast<float>(-1), calCount); // decending
    Sort32(tmpXVerticesLocal1, tmpXVerticesLocal, sortIdx.ReinterpretCast<uint32_t>(), boxesCount);
    GatherMask(tmpXVerticesLocal1, tmpXVerticesLocal1, src1Pattern, false, mask, { 1, repeatTimes, 8, 0 }, rsvdCnt);
    Gather(firstSortedIdxLocal, tmpXVerticesLocal1.ReinterpretCast<int32_t>(), gatherLocal.ReinterpretCast<uint32_t>(), 0u, boxesCount);

    // SelectFrontNineIdx
    BroadCast<int32_t, 2, 1, false>(idxLocal1, numValidLocal, broadCastDstShape1, broadCastSrcShape1);
    Cast(idxLocal1.ReinterpretCast<float>(), idxLocal1, RoundMode::CAST_NONE, calCount);
    Cast(tmpYVerticesLocal, sortIdx, RoundMode::CAST_NONE, calCount);
    Compare(tmpMaskLocal1, tmpYVerticesLocal, idxLocal1.ReinterpretCast<float>(), CMPMODE::LT, boxesCountAligned * VERTICES_ALIGNED);
    BroadCast<int32_t, 2, 1, false>(idxLocal1, firstSortedIdxLocal, broadCastDstShape1, broadCastSrcShape1);
    
    SetFlag<HardEvent::MTE3_V>(eventMTE3V);
    WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
    Select(SortedVerticesLocal.ReinterpretCast<float>(), tmpMaskLocal1, tmpXVerticesLocal1, idxLocal1.ReinterpretCast<float>(),
        SELMODE::VSEL_TENSOR_TENSOR_MODE, calCount);
}