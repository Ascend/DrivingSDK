#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

using namespace AscendC;
using namespace std;

constexpr uint32_t ALIGN_NUM = 8;
constexpr uint32_t FLOAT_SIZE = 4;
constexpr uint32_t DOUBLE_NUM = 2;
constexpr int32_t ONE_VALUE = 1;
constexpr int32_t ZERO_VALUE = 0;
constexpr float ZERO_FLOAT_VALUE = 0.0f;
constexpr float ONE_FLOAT_VALUE = 1.0f;

class GeometricKernelAttention {
public:
    __aicore__ inline GeometricKernelAttention()
    {}
    __aicore__ inline void Init(GM_ADDR value, GM_ADDR spatial_shapes, GM_ADDR level_start_index, GM_ADDR sampling_locations,
                                GM_ADDR attention_weights, GM_ADDR output, const GeometricKernelAttentionTilingData *tiling_data, TPipe* pipe)
    {
        ASSERT(GetBlockNum() != 0 && "Block Dim can not be Zero!");
        this->blockIndex = GetBlockIdx();
        this->_pipe = pipe;

        batchSize = tiling_data->batchSize;
        numKeys = tiling_data->numKeys;
        numHeads = tiling_data->numHeads;
        numQueries = tiling_data->numQueries;
        numLevels = tiling_data->numLevels;
        numPoints = tiling_data->numPoints;
        dim = tiling_data->dim;
        alignLevels = tiling_data->alignLevels;
        alignDim = tiling_data->alignDim;
        totalTaskNum = tiling_data->totalTaskNum;
        alignTaskNum = tiling_data->alignTaskNum;
        tailNum = tiling_data->tailNum;
        blockDim = tiling_data->blockDim;
        taskNumPerScore = tiling_data->taskNumPerScore;
        taskNumPerLcore = tiling_data->taskNumPerLcore;
        scoreNum = tiling_data->scoreNum;
        lcoreNum = tiling_data->lcoreNum;
        ubTotalSize = tiling_data->ubTotalSize;

        if (blockIndex < lcoreNum) {
            taskNumPerCore = taskNumPerLcore;
        } else {
            taskNumPerCore = taskNumPerScore;
        }

        uint32_t ubSizeForLoop = (static_cast<uint32_t>(ubTotalSize)) / ALIGN_NUM;
        taskNumPerLoop = taskNumPerCore;
        loopCount = 1;

        valueGM.SetGlobalBuffer((__gm__ DTYPE_VALUE *)value, batchSize * numKeys * numHeads * dim);
        spatialshapesGM.SetGlobalBuffer((__gm__ DTYPE_SPATIAL_SHAPES *)spatial_shapes, numLevels * DOUBLE_NUM);
        levelindexGM.SetGlobalBuffer((__gm__ DTYPE_LEVEL_START_INDEX *)level_start_index, numLevels);
        samplingGM.SetGlobalBuffer((__gm__ DTYPE_SAMPLING_LOCATIONS *)sampling_locations, batchSize * numQueries * numHeads * numLevels * numPoints * DOUBLE_NUM);
        attentionGM.SetGlobalBuffer((__gm__ DTYPE_ATTENTION_WEIGHTS *)attention_weights, batchSize * numQueries * numHeads * numLevels * numPoints);
        outputGM.SetGlobalBuffer((__gm__ DTYPE_OUTPUT *)output, batchSize * numQueries * numHeads * dim);

        this->_pipe->InitBuffer(ValueBuffer, alignDim * FLOAT_SIZE);
        this->_pipe->InitBuffer(AtomicAddBuffer, alignDim * FLOAT_SIZE);
        this->_pipe->InitBuffer(TmpBuffer, alignDim * FLOAT_SIZE);
        this->_pipe->InitBuffer(SpatialShapesBuffer, alignLevels * FLOAT_SIZE * DOUBLE_NUM);
        this->_pipe->InitBuffer(LevelStartIndexBuffer, alignLevels * FLOAT_SIZE);
        this->_pipe->InitBuffer(HeightLocationBuffer, alignLevels * numPoints * FLOAT_SIZE);
        this->_pipe->InitBuffer(WidthLocationBuffer, alignLevels * numPoints * FLOAT_SIZE);
        this->_pipe->InitBuffer(AttentionWeightBuffer, alignLevels * numPoints * FLOAT_SIZE);
        this->_pipe->InitBuffer(TmpSamplingLocationBuffer, alignLevels * numPoints * FLOAT_SIZE * DOUBLE_NUM);
        this->_pipe->InitBuffer(ClampBuffer, numPoints * alignLevels * FLOAT_SIZE);
        this->_pipe->InitBuffer(SamplingLevelBuffer, numPoints * alignLevels * FLOAT_SIZE);
        this->_pipe->InitBuffer(OutputBuffer, alignDim * FLOAT_SIZE);
    }

    __aicore__ inline void Process()
    {
        if (taskNumPerLoop > 0) {
            AllocLocalTensors();
            DuplicateAtomicAddTensor();
            
            for (uint32_t i = 0; i < loopCount; i++) {
                bool isLastLoop = IsLastLoopJudgment(i);
                int32_t loopStartIndex = LoopStartIndexCompute(i);

                LevelStartIndexCopyIn();
                SpatialShapesCopyIn();
                PipeBarrier<PIPE_ALL>();

                for (int32_t taskIndex = loopStartIndex; taskIndex < loopStartIndex + taskNumPerLoop; taskIndex++) {
                    Compute(taskIndex, isLastLoop);
                }
            }
        }
    }

private:
    __aicore__ inline void AllocLocalTensors()
    {
        SamplingLevelTensor = SamplingLevelBuffer.Get<float>();
        HeightLocationTensor = HeightLocationBuffer.Get<DTYPE_SAMPLING_LOCATIONS>();
        WidthLocationTensor = WidthLocationBuffer.Get<DTYPE_SAMPLING_LOCATIONS>();
        LevelStartIndexTensor = LevelStartIndexBuffer.Get<DTYPE_LEVEL_START_INDEX>();
        SpatialShapesTensor = SpatialShapesBuffer.Get<DTYPE_SPATIAL_SHAPES>();
        TmpSamplingLocationTensor = TmpSamplingLocationBuffer.Get<DTYPE_SAMPLING_LOCATIONS>();
        AttentionWeightTensor = AttentionWeightBuffer.Get<DTYPE_ATTENTION_WEIGHTS>();
        AtomicAddTensor = AtomicAddBuffer.Get<float>();
        ValueTensor = ValueBuffer.Get<DTYPE_VALUE>();
        OutputTensor = OutputBuffer.Get<DTYPE_OUTPUT>();
        ClampTensor = ClampBuffer.Get<float>();
        TmpTensor = TmpBuffer.Get<float>();
    }

    __aicore__ inline void DuplicateAtomicAddTensor()
    {
        Duplicate(AtomicAddTensor, ONE_FLOAT_VALUE, alignDim);
        for (int32_t idx = dim; idx < alignDim; idx++) {
            AtomicAddTensor.SetValue(idx, ZERO_FLOAT_VALUE);
        }
    }

    __aicore__ inline bool IsLastLoopJudgment(uint32_t loopIndex)
    {
        bool isLastLoop;
        if (loopIndex == loopCount - 1) {
            isLastLoop = true;
        } else {
            isLastLoop = false;
        }
        return isLastLoop;
    }

    __aicore__ inline int32_t LoopStartIndexCompute(uint32_t loopIndex)
    {
        int32_t loopStartIndex;
        if (blockIndex < lcoreNum) {
            loopStartIndex = blockIndex * taskNumPerCore + loopIndex * taskNumPerLoop;
        } else {
            loopStartIndex = lcoreNum * taskNumPerLcore + (blockIndex - lcoreNum) * taskNumPerScore + loopIndex * taskNumPerLoop;
        }
        return loopStartIndex;
    }

    __aicore__ inline void LevelStartIndexCopyIn()
    {
        DataCopy(LevelStartIndexTensor, levelindexGM[0], alignLevels);
    }

    __aicore__ inline void SpatialShapesCopyIn()
    {
        DataCopy(SpatialShapesTensor, spatialshapesGM[0], alignLevels * DOUBLE_NUM);
    }

    __aicore__ inline void SamplingLocationsCopyIn(int32_t index)
    {
        DataCopy(TmpSamplingLocationTensor, samplingGM[index], alignLevels * numPoints * DOUBLE_NUM);
        PipeBarrier<PIPE_ALL>();
        if (TmpSamplingLocationTensor.GetSize() >= 32) {
            GatherMaskLocations();
        } else {
            DuplicateLocations();
        }
    }

    __aicore__ inline void GatherMaskLocations()
    {
        uint8_t src0BlockStride = 1;
        uint8_t src1RepeatStride = 0;
        uint16_t src0RepeatStride = 8;
        uint16_t repeatTimes = TmpSamplingLocationTensor.GetSize() / (ALIGN_NUM * FLOAT_SIZE);
        uint32_t height_pattern = 2;
        uint32_t width_pattern = 1;
        uint32_t mask = 0;
        uint64_t rsvdCnt = 0;
        
        GatherMask(HeightLocationTensor, TmpSamplingLocationTensor, height_pattern, false, mask, {src0BlockStride, repeatTimes, src0RepeatStride, src1RepeatStride}, rsvdCnt);
        GatherMask(WidthLocationTensor, TmpSamplingLocationTensor, width_pattern, false, mask, {src0BlockStride, repeatTimes, src0RepeatStride, src1RepeatStride}, rsvdCnt);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DuplicateLocations()
    {
        for (int32_t i = 0; i < (numLevels * numPoints * DOUBLE_NUM); i++) {
            if (static_cast<int32_t>(i % DOUBLE_NUM) == 0) {
                WidthLocationTensor.SetValue(static_cast<int32_t>(i / DOUBLE_NUM), TmpSamplingLocationTensor.GetValue(i));
            } else {
                HeightLocationTensor.SetValue(static_cast<int32_t>(i / DOUBLE_NUM), TmpSamplingLocationTensor.GetValue(i));
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AttentionWeightCopyIn(int32_t index)
    {
        DataCopy(AttentionWeightTensor, attentionGM[index], alignLevels * numPoints);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void MultiHeadSamplingCopyIn(int32_t copyIndex)
    {
        SamplingLocationsCopyIn(copyIndex * DOUBLE_NUM);
        AttentionWeightCopyIn(copyIndex);
    }

    __aicore__ inline void SamplingLocationsClamp(bool heightclamp, int32_t threshold)
    {
        if (heightclamp == true) {
            ClampMax(ClampTensor, HeightLocationTensor, static_cast<float>(threshold - ONE_VALUE), numPoints * alignLevels);
            ClampMin(HeightLocationTensor, ClampTensor, ZERO_FLOAT_VALUE, numPoints * alignLevels);
        } else {
            ClampMax(ClampTensor, WidthLocationTensor, static_cast<float>(threshold - ONE_VALUE), numPoints * alignLevels);
            ClampMin(WidthLocationTensor, ClampTensor, ZERO_FLOAT_VALUE, numPoints * alignLevels);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SingleValueCopyIn(int32_t copyIndex)
    {
        DataCopy(ValueTensor, valueGM[copyIndex], alignDim);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void SingleValueCopyOut(int32_t copyIndex)
    {
        Mul(OutputTensor, OutputTensor, AtomicAddTensor, alignDim);
        PipeBarrier<PIPE_ALL>();

        SetAtomicAdd<float>();
        DataCopy(outputGM[copyIndex], OutputTensor, alignDim);
        SetAtomicNone();
    }

    __aicore__ inline void MultiScaleKernelAttnSampling(int32_t batchIndex, int32_t headIndex, int32_t queryIndex)
    {
        Duplicate(OutputTensor, ZERO_FLOAT_VALUE, alignDim);

        for (int32_t levelIndex = 0; levelIndex < numLevels; levelIndex++) {
            SamplingLocationsClamp(true, SpatialShapesTensor.GetValue(levelIndex * DOUBLE_NUM));
            SamplingLocationsClamp(false, SpatialShapesTensor.GetValue(levelIndex * DOUBLE_NUM + ONE_VALUE));
            int32_t level_width = SpatialShapesTensor.GetValue(levelIndex * DOUBLE_NUM + ONE_VALUE);

            for (int32_t pointIndex = 0; pointIndex < numPoints; pointIndex++) {
                float pointAttentionWeight = AttentionWeightTensor.GetValue(levelIndex * numPoints + pointIndex);
                int32_t wLocation = WidthLocationTensor.GetValue(levelIndex * numPoints + pointIndex);
                int32_t hLocation = HeightLocationTensor.GetValue(levelIndex * numPoints + pointIndex);
                int32_t levelStartId = LevelStartIndexTensor.GetValue(levelIndex);
                int32_t valueCopyinIndex = batchIndex * numKeys * numHeads * dim + headIndex * numKeys * dim + (levelStartId + hLocation * level_width + wLocation) * dim;

                PipeBarrier<PIPE_ALL>();
                SingleValueCopyIn(valueCopyinIndex);
                Muls(ValueTensor, ValueTensor, pointAttentionWeight, dim);
                PipeBarrier<PIPE_V>();
                Add(OutputTensor, ValueTensor, OutputTensor, dim);
            }
        }
        int32_t valueCopyoutIndex = batchIndex * numQueries * numHeads * dim + queryIndex * numHeads * dim + headIndex * dim;
        SingleValueCopyOut(valueCopyoutIndex);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Compute(int32_t sampling_index, bool isLastLoop)
    {
        int32_t headIndex = sampling_index % numHeads;
        int32_t queryIndex = (sampling_index / numHeads) % numQueries;
        int32_t batchIndex = sampling_index / numHeads / numQueries;
        int32_t copyIndex = (batchIndex * numQueries * numHeads + queryIndex * numHeads + headIndex) * numLevels * numPoints;

        MultiHeadSamplingCopyIn(copyIndex);
        MultiScaleKernelAttnSampling(batchIndex, headIndex, queryIndex);
    }

private:
    TPipe *_pipe;
    TBuf <TPosition::VECCALC> ValueBuffer, LevelStartIndexBuffer, SpatialShapesBuffer, HeightLocationBuffer, WidthLocationBuffer, AttentionWeightBuffer;
    TBuf <TPosition::VECCALC> TmpSamplingLocationBuffer, SamplingLevelBuffer, ClampBuffer, AtomicAddBuffer, TmpBuffer;
    TBuf <TPosition::VECCALC> OutputBuffer;
    LocalTensor<int32_t> LevelStartIndexTensor, SpatialShapesTensor;
    LocalTensor<float> HeightLocationTensor, WidthLocationTensor, TmpSamplingLocationTensor, AttentionWeightTensor;
    LocalTensor<float> SamplingLevelTensor, ClampTensor, AtomicAddTensor, TmpTensor;
    LocalTensor<float> ValueTensor, OutputTensor;

    GlobalTensor<DTYPE_VALUE> valueGM;
    GlobalTensor<DTYPE_SPATIAL_SHAPES> spatialshapesGM;
    GlobalTensor<DTYPE_LEVEL_START_INDEX> levelindexGM;
    GlobalTensor<DTYPE_SAMPLING_LOCATIONS> samplingGM;
    GlobalTensor<DTYPE_ATTENTION_WEIGHTS> attentionGM;
    GlobalTensor<DTYPE_OUTPUT> outputGM;

    uint64_t blockIndex;
    int32_t batchSize, numKeys, numHeads, numQueries, numLevels, numPoints, dim, alignLevels, alignDim, totalTaskNum, alignTaskNum, tailNum;
    int32_t sampling_loop_count;
    uint32_t blockDim, taskNumPerScore, taskNumPerLcore, scoreNum, lcoreNum, taskNumPerCore;
    uint32_t ubSizeForLoop, loopCount, taskNumPerLoop, taskNumPerLoop_limit;
    uint64_t ubTotalSize;
};

extern "C" __global__ __aicore__ void geometric_kernel_attention(GM_ADDR value, GM_ADDR spatial_shapes, GM_ADDR level_start_index, GM_ADDR sampling_locations,
                                                                GM_ADDR attention_weights, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        GeometricKernelAttention op;
        op.Init(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, output, &tiling_data, &pipe);
        op.Process();
    }
}