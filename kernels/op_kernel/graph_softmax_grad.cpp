#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

using namespace AscendC;
using namespace std;

constexpr int32_t ALIGN_NUM = 8;
constexpr int32_t FLOAT_SIZE = 4;
constexpr int32_t DOUBLE_NUM = 2;
constexpr int32_t BUFFER_NUM = 5;
constexpr int32_t ONE_VALUE = 1;
constexpr int32_t ZERO_VALUE = 0;
constexpr int32_t SRC_SHAPE_DIM = 8;
constexpr float ZERO_FLOAT_VALUE = 0.0f;
constexpr float ONE_FLOAT_VALUE = 1.0f;
constexpr float UB_RATIO = 0.8f;

class GraphSoftmaxGrad {
public:
    __aicore__ inline GraphSoftmaxGrad()
    {}
    __aicore__ inline void Init(GM_ADDR index, GM_ADDR softmax_output, GM_ADDR grad_output, GM_ADDR reduce_sum, GM_ADDR src_grad, const GraphSoftmaxGradTilingData *tiling_data, TPipe* pipe)
    {
        ASSERT(GetBlockNum() != 0 && "Block Dim can not be Zero!");
        this->blockIndex = GetBlockIdx();
        this->_pipe = pipe;

        GetTilingData(tiling_data);
        uint64_t bufferSize = taskNumPerLoop * SRC_SHAPE_DIM * FLOAT_SIZE;
        uint64_t castEdgeNum = static_cast<uint64_t>(edgeNum);
        uint64_t gmSize = castEdgeNum * SRC_SHAPE_DIM;

        indexGM.SetGlobalBuffer((__gm__ DTYPE_INDEX *)index, castEdgeNum);
        softmaxGM.SetGlobalBuffer((__gm__ DTYPE_SOFTMAX_OUTPUT *)softmax_output, gmSize);
        gradGM.SetGlobalBuffer((__gm__ DTYPE_GRAD_OUTPUT *)grad_output, gmSize);
        srcgradGM.SetGlobalBuffer((__gm__ DTYPE_SRC_GRAD *)src_grad, gmSize);
        reducesumGM.SetGlobalBuffer((__gm__ DTYPE_REDUCE_SUM *)reduce_sum, nodeNum * SRC_SHAPE_DIM);

        this->_pipe->InitBuffer(SoftmaxOutputBuffer, bufferSize);
        this->_pipe->InitBuffer(IndexTensorBuffer, taskNumPerLoop * FLOAT_SIZE);
        this->_pipe->InitBuffer(GradOutputBuffer, bufferSize);
        this->_pipe->InitBuffer(SrcGradTensorBuffer, bufferSize);
        this->_pipe->InitBuffer(TmpIndexSelectBuffer, bufferSize);
    }

    __aicore__ inline void Process()
    {
        AllocLocalTensors();
        if (taskLoop == 1) {
            SingleLoopComputing();
        } else {
            MultiLoopComputing();
        }
    }

private:

    __aicore__ inline void GetTilingData(const GraphSoftmaxGradTilingData *tiling_data)
    {
        edgeNum = tiling_data->edgeNum;
        alignTaskNum = tiling_data->alignTaskNum;
        tailNum = tiling_data->tailNum;
        nodeNum = tiling_data->nodeNum;
        taskNumPerLoop = tiling_data->taskNumPerLoop;
        taskLoop = tiling_data->taskLoop;
        blockDim = tiling_data->blockDim;
        tailCoreNum = tiling_data->tailCoreNum;
        taskNumPerCore = tiling_data->taskNumPerCore;
        ubTotalSize = tiling_data->ubTotalSize;
    }

    __aicore__ inline uint64_t GetCopyIndex(uint32_t taskLoopIndex)
    {
        uint64_t copyIndex = blockIndex * taskNumPerCore + taskLoopIndex * taskNumPerLoop;
        if (blockIndex >= tailCoreNum) {
            copyIndex -= (blockIndex - tailCoreNum);
        }
        return copyIndex;
    }

    __aicore__ inline int32_t GetTaskNum(uint32_t taskLoopIndex)
    {
        int32_t taskNumPerCurLoop;
        if (taskLoopIndex == taskLoop - 1) {
            taskNumPerCurLoop = taskNumPerCore - taskLoopIndex * taskNumPerLoop;
            if (blockIndex >= tailCoreNum) {
                taskNumPerCurLoop -= 1;
            }
        } else {
            taskNumPerCurLoop = taskNumPerLoop;
        }
        return taskNumPerCurLoop;
    }

    __aicore__ inline void SingleLoopComputing()
    {
        uint64_t copyIndex = GetCopyIndex(ZERO_VALUE);
        int32_t taskNumPerCurLoop = GetTaskNum(ZERO_VALUE);
        CopyIn(copyIndex, taskNumPerCurLoop);
        ReduceSum(taskNumPerCurLoop);
        SyncAll();
        SingleLoopIndexSelect(taskNumPerCurLoop);
        CopyOut(copyIndex, taskNumPerCurLoop);
    }

    __aicore__ inline void MultiLoopComputing()
    {
        for (uint32_t taskLoopIndex = 0; taskLoopIndex < taskLoop; taskLoopIndex++) {
            uint64_t copyIndex = GetCopyIndex(taskLoopIndex);
            int32_t taskNumPerCurLoop = GetTaskNum(taskLoopIndex);
            CopyIn(copyIndex, taskNumPerCurLoop);
            ReduceSum(taskNumPerCurLoop);
        }
        SyncAll();

        for (uint32_t taskLoopIndex = 0; taskLoopIndex < taskLoop; taskLoopIndex++) {
            uint64_t copyIndex = GetCopyIndex(taskLoopIndex);
            int32_t taskNumPerCurLoop = GetTaskNum(taskLoopIndex);
            int32_t IndexCopyLength = (taskNumPerCurLoop + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
            DataCopy(IndexTensor, indexGM[copyIndex], IndexCopyLength);
            MultiLoopIndexSelect(copyIndex, taskNumPerCurLoop);
            CopyOut(copyIndex, taskNumPerCurLoop);
        }
    }

    __aicore__ inline void AllocLocalTensors()
    {
        IndexTensor = IndexTensorBuffer.Get<DTYPE_INDEX>();
        SoftmaxOutputTensor = SoftmaxOutputBuffer.Get<DTYPE_SOFTMAX_OUTPUT>();
        GradOutputTensor = GradOutputBuffer.Get<DTYPE_GRAD_OUTPUT>();
        SrcGradTensor = SrcGradTensorBuffer.Get<DTYPE_SRC_GRAD>();
        TmpIndexSelectTensor = TmpIndexSelectBuffer.Get<DTYPE_SRC_GRAD>();

        Duplicate(GradOutputTensor, ZERO_FLOAT_VALUE, taskNumPerLoop * SRC_SHAPE_DIM);
        Duplicate(TmpIndexSelectTensor, ZERO_FLOAT_VALUE, taskNumPerLoop * SRC_SHAPE_DIM);
    }

    __aicore__ inline void CopyIn(uint64_t copyIndex, int32_t copyLength)
    {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        int32_t IndexCopyLength = (copyLength + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
        DataCopy(IndexTensor, indexGM[copyIndex], IndexCopyLength);
        DataCopy(SoftmaxOutputTensor, softmaxGM[copyIndex * SRC_SHAPE_DIM], copyLength * SRC_SHAPE_DIM);
        DataCopy(GradOutputTensor, gradGM[copyIndex * SRC_SHAPE_DIM], copyLength * SRC_SHAPE_DIM);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        Mul(GradOutputTensor, SoftmaxOutputTensor, GradOutputTensor, GradOutputTensor.GetSize());
    }

    __aicore__ inline void ReduceSum(int32_t egTotalNum)
    {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        SetAtomicAdd<float>();
        for (int32_t egIndex = 0; egIndex < egTotalNum; egIndex++) {
            int32_t group = IndexTensor.GetValue(egIndex);
            DataCopy(reducesumGM[group * SRC_SHAPE_DIM], GradOutputTensor[egIndex * SRC_SHAPE_DIM], SRC_SHAPE_DIM);
        }
        SetAtomicNone();

        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }

    __aicore__ inline void SingleLoopIndexSelect(int32_t egTotalNum)
    {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        for (int32_t egIndex = 0; egIndex < egTotalNum; egIndex++) {
            int32_t group = IndexTensor.GetValue(egIndex);
            DataCopy(TmpIndexSelectTensor[egIndex * SRC_SHAPE_DIM], reducesumGM[group * SRC_SHAPE_DIM], SRC_SHAPE_DIM);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        Mul(TmpIndexSelectTensor, TmpIndexSelectTensor, SoftmaxOutputTensor, SoftmaxOutputTensor.GetSize());
        Sub(SrcGradTensor, GradOutputTensor, TmpIndexSelectTensor, TmpIndexSelectTensor.GetSize());
    }

    __aicore__ inline void MultiLoopIndexSelect(uint64_t copyIndex, int32_t egTotalNum)
    {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        for (int32_t egIndex = 0; egIndex < egTotalNum; egIndex++) {
            int32_t group = IndexTensor.GetValue(egIndex);
            DataCopy(TmpIndexSelectTensor[egIndex * SRC_SHAPE_DIM], reducesumGM[group * SRC_SHAPE_DIM], SRC_SHAPE_DIM);
        }

        DataCopy(SoftmaxOutputTensor, softmaxGM[copyIndex * SRC_SHAPE_DIM], egTotalNum * SRC_SHAPE_DIM);
        DataCopy(GradOutputTensor, gradGM[copyIndex * SRC_SHAPE_DIM], egTotalNum * SRC_SHAPE_DIM);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        Mul(GradOutputTensor, SoftmaxOutputTensor, GradOutputTensor, GradOutputTensor.GetSize());
        Mul(TmpIndexSelectTensor, TmpIndexSelectTensor, SoftmaxOutputTensor, SoftmaxOutputTensor.GetSize());
        Sub(SrcGradTensor, GradOutputTensor, TmpIndexSelectTensor, TmpIndexSelectTensor.GetSize());
    }

    __aicore__ inline void CopyOut(uint64_t copyIndex, int32_t copyLength)
    {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        DataCopy(srcgradGM[copyIndex * SRC_SHAPE_DIM], SrcGradTensor, copyLength * SRC_SHAPE_DIM);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }

private:
    TPipe *_pipe;
    TBuf <TPosition::VECCALC> IndexTensorBuffer, SoftmaxOutputBuffer, GradOutputBuffer;
    TBuf <TPosition::VECCALC> SrcGradTensorBuffer;
    TBuf <TPosition::VECCALC> TmpIndexSelectBuffer;

    LocalTensor<int32_t> IndexTensor;
    LocalTensor<float> SoftmaxOutputTensor, GradOutputTensor;
    LocalTensor<float> SrcGradTensor;
    LocalTensor<float> TmpIndexSelectTensor;

    GlobalTensor<DTYPE_INDEX> indexGM;
    GlobalTensor<DTYPE_SOFTMAX_OUTPUT> softmaxGM;
    GlobalTensor<DTYPE_GRAD_OUTPUT> gradGM;
    GlobalTensor<DTYPE_SRC_GRAD> srcgradGM;
    GlobalTensor<DTYPE_REDUCE_SUM> reducesumGM;

    int32_t edgeNum, nodeNum, alignTaskNum, tailNum, taskStartIndex, taskLoop, taskNumPerLoop, tailCoreNum;
    uint32_t blockDim, taskNumPerCore;
    uint64_t blockIndex, ubTotalSize;
};

extern "C" __global__ __aicore__ void graph_softmax_grad(GM_ADDR index, GM_ADDR softmax_output, GM_ADDR grad_output, GM_ADDR reduce_sum, GM_ADDR src_grad, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        GraphSoftmaxGrad op;
        op.Init(index, softmax_output, grad_output, reduce_sum, src_grad, &tiling_data, &pipe);
        op.Process();
    }
}