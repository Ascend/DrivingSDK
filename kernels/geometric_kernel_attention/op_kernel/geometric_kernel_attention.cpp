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
            taskStartIndex = blockIndex * taskNumPerCore;
        } else {
            taskNumPerCore = taskNumPerScore;
            taskStartIndex = lcoreNum * taskNumPerLcore + (blockIndex - lcoreNum) * taskNumPerCore;
        }

        uint64_t castBatchSize = static_cast<uint64_t>(batchSize);
        uint64_t castNumLevel = static_cast<uint64_t>(numLevels);
        valueGM.SetGlobalBuffer((__gm__ DTYPE_VALUE *)value, castBatchSize * numKeys * numHeads * dim);
        spatialshapesGM.SetGlobalBuffer((__gm__ DTYPE_SPATIAL_SHAPES *)spatial_shapes, castNumLevel * DOUBLE_NUM);
        levelindexGM.SetGlobalBuffer((__gm__ DTYPE_LEVEL_START_INDEX *)level_start_index, castNumLevel);
        samplingGM.SetGlobalBuffer((__gm__ DTYPE_SAMPLING_LOCATIONS *)sampling_locations, castBatchSize * numQueries * numHeads * numLevels * numPoints * DOUBLE_NUM);
        attentionGM.SetGlobalBuffer((__gm__ DTYPE_ATTENTION_WEIGHTS *)attention_weights, castBatchSize * numQueries * numHeads * numLevels * numPoints);
        outputGM.SetGlobalBuffer((__gm__ DTYPE_OUTPUT *)output, castBatchSize * numQueries * numHeads * dim);

        uint64_t dimBufferSize = alignDim * FLOAT_SIZE;
        this->_pipe->InitBuffer(ValueBuffer, numPoints * dimBufferSize);
        this->_pipe->InitBuffer(AtomicAddBuffer, dimBufferSize);
        this->_pipe->InitBuffer(SpatialShapesBuffer, alignLevels * FLOAT_SIZE * DOUBLE_NUM);
        this->_pipe->InitBuffer(LevelStartIndexBuffer, alignLevels * FLOAT_SIZE);
        this->_pipe->InitBuffer(AttentionWeightBuffer, alignLevels * numPoints * FLOAT_SIZE);
        this->_pipe->InitBuffer(TmpSamplingLocationBuffer, alignLevels * numPoints * FLOAT_SIZE * DOUBLE_NUM);
        this->_pipe->InitBuffer(OutputBuffer, dimBufferSize);
    }

    __aicore__ inline void Process()
    {
        if (taskNumPerCore > 0) {
            AllocLocalTensors();
            DuplicateAtomicAddTensor();
            LevelStartIndexCopyIn();
            SpatialShapesCopyIn();
            
            for (int32_t taskIndex = taskStartIndex; taskIndex < taskStartIndex + taskNumPerCore; taskIndex++) {
                Compute(taskIndex);
            }
        }
    }

private:
    __aicore__ inline void AllocLocalTensors()
    {
        LevelStartIndexTensor = LevelStartIndexBuffer.Get<DTYPE_LEVEL_START_INDEX>();
        SpatialShapesTensor = SpatialShapesBuffer.Get<DTYPE_SPATIAL_SHAPES>();
        TmpSamplingLocationTensor = TmpSamplingLocationBuffer.Get<DTYPE_SAMPLING_LOCATIONS>();
        AttentionWeightTensor = AttentionWeightBuffer.Get<DTYPE_ATTENTION_WEIGHTS>();
        AtomicAddTensor = AtomicAddBuffer.Get<float>();
        ValueTensor = ValueBuffer.Get<DTYPE_VALUE>();
        OutputTensor = OutputBuffer.Get<DTYPE_OUTPUT>();
    }

    __aicore__ inline void DuplicateAtomicAddTensor()
    {
        Duplicate(AtomicAddTensor, ONE_FLOAT_VALUE, alignDim);
        for (int32_t idx = dim; idx < alignDim; idx++) {
            AtomicAddTensor.SetValue(idx, ZERO_FLOAT_VALUE);
        }
    }

    __aicore__ inline void LevelStartIndexCopyIn()
    {
        DataCopy(LevelStartIndexTensor, levelindexGM[0], alignLevels);
    }

    __aicore__ inline void SpatialShapesCopyIn()
    {
        DataCopy(SpatialShapesTensor, spatialshapesGM[0], alignLevels * DOUBLE_NUM);
    }

    __aicore__ inline void MultiHeadSamplingCopyIn(uint64_t copyIndex)
    {
        DataCopy(TmpSamplingLocationTensor, samplingGM[copyIndex * DOUBLE_NUM], alignLevels * numPoints * DOUBLE_NUM);
        DataCopy(AttentionWeightTensor, attentionGM[copyIndex], alignLevels * numPoints);
    }

    __aicore__ inline int32_t ScalarClamp(int32_t scalarValue, int32_t minValue, int32_t maxValue)
    {
        if (scalarValue < minValue) {
            scalarValue = minValue;
        }
        if (scalarValue > maxValue) {
            scalarValue = maxValue;
        }
        return scalarValue;
    }

    __aicore__ inline void SingleValueCopyOut(int32_t copyIndex)
    {
        SetAtomicAdd<float>();
        DataCopy(outputGM[static_cast<uint64_t>(copyIndex)], OutputTensor, alignDim);
        SetAtomicNone();
    }

    __aicore__ inline void MultiScaleKernelAttnSampling(int32_t valueCopyInPreIndex, int32_t valueCopyoutIndex)
    {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        Duplicate(OutputTensor, ZERO_FLOAT_VALUE, alignDim);

        for (int32_t levelIndex = 0; levelIndex < numLevels; levelIndex++) {
            int32_t levelWidth = SpatialShapesTensor.GetValue(levelIndex * DOUBLE_NUM + ONE_VALUE);
            int32_t levelHeight = SpatialShapesTensor.GetValue(levelIndex * DOUBLE_NUM);
            int32_t levelStartId = LevelStartIndexTensor.GetValue(levelIndex);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            for (int32_t pointIndex = 0; pointIndex < numPoints; pointIndex++) {
                int32_t wLocation = TmpSamplingLocationTensor.GetValue((levelIndex * numPoints + pointIndex) * 2);
                int32_t hLocation = TmpSamplingLocationTensor.GetValue((levelIndex * numPoints + pointIndex) * 2 + 1);
                wLocation = ScalarClamp(wLocation, 0, levelWidth - 1);
                hLocation = ScalarClamp(hLocation, 0, levelHeight - 1);
                int32_t valueCopyinIndex = (valueCopyInPreIndex + (levelStartId + hLocation * levelWidth + wLocation)) * dim;
                DataCopy(ValueTensor[pointIndex * alignDim], valueGM[static_cast<uint64_t>(valueCopyinIndex)], alignDim);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                float pointAttentionWeight = AttentionWeightTensor.GetValue(levelIndex * numPoints + pointIndex);
                Axpy(OutputTensor, ValueTensor[pointIndex * alignDim], pointAttentionWeight, dim);
            }
        }
        Mul(OutputTensor, OutputTensor, AtomicAddTensor, alignDim);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        SingleValueCopyOut(valueCopyoutIndex);
    }

    __aicore__ inline void Compute(int32_t samplingIndex)
    {
        int32_t headIndex = samplingIndex % numHeads;
        int32_t queryIndex = (samplingIndex / numHeads) % numQueries;
        int32_t batchIndex = samplingIndex / numHeads / numQueries;

        int32_t copyIndex = ((batchIndex * numQueries + queryIndex) * numHeads + headIndex) * numLevels * numPoints;
        int32_t valueCopyInPreIndex = batchIndex * numKeys * numHeads + headIndex * numKeys;
        int32_t valueCopyoutIndex = (batchIndex * numQueries * numHeads + queryIndex * numHeads + headIndex) * dim;

        MultiHeadSamplingCopyIn(static_cast<uint64_t>(copyIndex));
        MultiScaleKernelAttnSampling(valueCopyInPreIndex, valueCopyoutIndex);
    }

private:
    TPipe *_pipe;
    TBuf <TPosition::VECCALC> ValueBuffer, LevelStartIndexBuffer, SpatialShapesBuffer, AttentionWeightBuffer;
    TBuf <TPosition::VECCALC> TmpSamplingLocationBuffer, AtomicAddBuffer;
    TBuf <TPosition::VECCALC> OutputBuffer;
    LocalTensor<int32_t> LevelStartIndexTensor, SpatialShapesTensor;
    LocalTensor<float> TmpSamplingLocationTensor, AttentionWeightTensor, AttentionWeightDuplicateTensor;
    LocalTensor<float> AtomicAddTensor;
    LocalTensor<float> ValueTensor, OutputTensor;

    GlobalTensor<DTYPE_VALUE> valueGM;
    GlobalTensor<DTYPE_SPATIAL_SHAPES> spatialshapesGM;
    GlobalTensor<DTYPE_LEVEL_START_INDEX> levelindexGM;
    GlobalTensor<DTYPE_SAMPLING_LOCATIONS> samplingGM;
    GlobalTensor<DTYPE_ATTENTION_WEIGHTS> attentionGM;
    GlobalTensor<DTYPE_OUTPUT> outputGM;

    uint64_t blockIndex;
    int32_t batchSize, numKeys, numHeads, numQueries, numLevels, numPoints, dim, alignLevels, alignDim, totalTaskNum, alignTaskNum, tailNum;
    int32_t taskStartIndex;
    uint32_t blockDim, taskNumPerScore, taskNumPerLcore, scoreNum, lcoreNum, taskNumPerCore;
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