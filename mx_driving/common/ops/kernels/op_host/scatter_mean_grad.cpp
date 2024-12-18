/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "scatter_mean_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace std;
using namespace AscendC;

namespace optiling {
constexpr uint32_t WORKSPACE_16MBYTE_SIZE = 16 * 1024 * 1024;
constexpr int64_t DATA_SMALL_MODE = 1;
constexpr int64_t NOT_BROAD_LINE_MODE = 2;
constexpr int64_t DATA_LARGE_MODE = 3;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint64_t RESERVE_SAPCE = 4 * 1024;
constexpr uint32_t FLOAT_DTYPE_BYTES = 4;

constexpr uint64_t MAX_OUT_LINE =  16000;
constexpr uint64_t MAX_DEAL_NUM =  2048;
constexpr uint64_t LEAST_LINE_EACH_TASK = 4;
constexpr uint64_t BUFFER_NUM = 2;

constexpr uint64_t INDICES_ONCE_DATANUM = 2048;
constexpr uint64_t GRADOUT_UB_NUM = 2;
constexpr uint64_t INDICES_UB_NUM = 2;

class ScatterMeanGradTiling {
public:
    explicit ScatterMeanGradTiling(gert::TilingContext* context) : context(context){};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint();
private:
    void SetModeNoTail(int32_t gradDims, int32_t indexDims, uint32_t coreNum);
    void SetModeLine(int32_t gradDims, int32_t indexDims, uint32_t coreNum);
    void SetHeadNumForTask(uint64_t headMaxTask, uint32_t coreNum);
    void SetUbSize(uint64_t headIndicesSize);
    ScatterMeanGradTilingData TilingData;
    gert::TilingContext* context = nullptr;
    uint64_t paramsPre = 1;
    uint64_t dimRange = 1;
    uint64_t dimRangeOut = 1;
    uint64_t paramsPro = 1;
    uint64_t tail = 1;
    int32_t dim = 0;
    uint32_t coreUsed = 1;

    uint64_t ubSize = 192 * 1024;
    uint64_t gradInUbSize = 1;
    uint64_t indexUbSize = 1;
    uint64_t gradOutUbSize = 1;
    uint64_t indexSumUbSize = 1;
    uint64_t gradInNum = 1;
    uint64_t indexNum = 1;
    uint64_t gradOutNum = 1;
    uint64_t countNum = 1;

    uint64_t headTaskSmall = 1;
    uint64_t taskNumSmall = 1;
    uint64_t headLastTaskSmall = 1;
    uint64_t headTaskBig = 1;
    uint64_t taskNumBig = 1;
    uint64_t headLastTaskBig = 1;
    uint64_t bigCoreNum = 1;
    uint64_t taskEachHead = 1;
    uint64_t tilingMode = 0;

    uint32_t gradDsize = 4;
    uint32_t paramsNumPerBlock = 1;
    uint32_t indexNumPerBlock = 8;
    uint32_t indexDsize = 4;
};

uint64_t CeilValue(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return 0;
    }
    return ((a - 1) / b + 1) * b;
}

static uint64_t GetCeilInt(uint64_t value1, uint64_t value2)
{
    if (value2 == 0) {
        return value1;
    }
    return (value1 + value2 - 1) / value2;
}

static void ComputeTaskNumForeLine(uint64_t ubOutNum, uint64_t outLineEachCore, uint64_t *taskNum, uint64_t *taskEachLine, uint64_t *taskLastLine)
{
    if (outLineEachCore <= ubOutNum) {
        *taskNum = 1;
        *taskEachLine = outLineEachCore;
        *taskLastLine = outLineEachCore;
    } else {
        uint64_t taskNumTemp = GetCeilInt(outLineEachCore, ubOutNum);
        *taskNum = taskNumTemp;
        *taskEachLine = ubOutNum;
        *taskLastLine = outLineEachCore - ubOutNum * (taskNumTemp - 1);
    }
}

void ScatterMeanGradTiling::SetUbSize(uint64_t headIndicesSize)
{
    if (headIndicesSize > INDICES_ONCE_DATANUM) {
        indexUbSize = INDICES_ONCE_DATANUM;
    } else {
        indexUbSize = CeilValue(headIndicesSize, indexNumPerBlock);
    }
    uint64_t ubAvailableSize = ubSize - indexUbSize * indexDsize * INDICES_UB_NUM;
    gradOutUbSize = CeilValue(ubAvailableSize / GRADOUT_UB_NUM / gradDsize, paramsNumPerBlock);
    return;
}

void ScatterMeanGradTiling::SetHeadNumForTask(uint64_t headMaxTask, uint32_t coreNum)
{
    uint64_t headBigCore = GetCeilInt(paramsPre, coreNum);
    uint64_t headSmallCore = headBigCore - 1;
    bigCoreNum = paramsPre - headSmallCore * coreNum;

    headTaskSmall = std::min(headMaxTask, headSmallCore);
    taskNumSmall = GetCeilInt(headSmallCore, headTaskSmall);
    headLastTaskSmall = headSmallCore - (taskNumSmall - 1) * headTaskSmall;

    headTaskBig = std::min(headMaxTask, headBigCore);
    taskNumBig = GetCeilInt(headBigCore, headTaskBig);
    headLastTaskBig = headBigCore - (taskNumBig - 1) * headTaskBig;
}

void ScatterMeanGradTiling::SetModeNoTail(int32_t gradDims, int32_t indexDims, uint32_t coreNum)
{
    if (coreNum == 0 || gradDims == 0 || indexDims == 0) {
        return;
    }
    
    uint64_t headOutSize = dimRangeOut * paramsPro;
    uint64_t headIndicesSize = dimRange * paramsPro;

    uint64_t ubBytesforHead = headOutSize * gradDsize * GRADOUT_UB_NUM + headIndicesSize * indexDsize * INDICES_UB_NUM;
    if (ubBytesforHead < ubSize) {
        context->SetTilingKey(DATA_SMALL_MODE);
        tilingMode = 0;
        auto headMaxTask = ubSize / ubBytesforHead;
        SetHeadNumForTask(headMaxTask, coreNum);
        gradOutUbSize = headTaskBig * headOutSize;
        indexUbSize = headTaskBig * headIndicesSize;
    } else {
        SetUbSize(headIndicesSize);
        if (gradOutUbSize > headOutSize)  {
            context->SetTilingKey(DATA_SMALL_MODE);
            tilingMode = 1;
            auto headMaxTask = gradOutUbSize / headOutSize;
            SetHeadNumForTask(headMaxTask, coreNum);
            gradOutUbSize = headTaskBig * headOutSize;
            indexUbSize = std::min((ubSize - gradOutUbSize * GRADOUT_UB_NUM * gradDsize) / INDICES_UB_NUM / indexDsize, headIndicesSize);
            indexUbSize = CeilValue(indexUbSize, indexNumPerBlock);
        } else {
            context->SetTilingKey(DATA_LARGE_MODE);
            tilingMode = DATA_LARGE_MODE;
            taskEachHead = GetCeilInt(headOutSize, gradOutUbSize);
            auto taskNum = paramsPre * taskEachHead;
            taskNumSmall = taskNum / coreNum;
            taskNumBig = taskNumSmall + 1;
            bigCoreNum = taskNum - taskNumSmall * coreNum;
        }
    }
    coreUsed = taskNumSmall == 0 ? bigCoreNum : coreNum;

    TilingData.set_headTaskSmall(headTaskSmall);
    TilingData.set_taskNumSmall(taskNumSmall);
    TilingData.set_headLastTaskSmall(headLastTaskSmall);
    TilingData.set_headTaskBig(headTaskBig);
    TilingData.set_taskNumBig(taskNumBig);
    TilingData.set_headLastTaskBig(headLastTaskBig);
    TilingData.set_taskEachHead(taskEachHead);
}

void ScatterMeanGradTiling::SetModeLine(int32_t gradDims, int32_t indexDims, uint32_t coreNum)
{
    if (coreNum == 0 || gradDims == 0 || indexDims == 0) {
        return;
    }
    context->SetTilingKey(NOT_BROAD_LINE_MODE);

    uint64_t dataLineSmallCore = 1;
    uint64_t dataLineBigCore = 1;

    auto body = paramsPro / tail;
    uint64_t dataLine = dimRange * paramsPre * (paramsPro / tail);
    
    if (dataLine <= 2 * LEAST_LINE_EACH_TASK) {
        coreUsed = 1;
        dataLineBigCore = dataLine;
        bigCoreNum = 1;
    } else {
        dataLineBigCore = std::max(GetCeilInt(dataLine, coreNum), LEAST_LINE_EACH_TASK);
        dataLineSmallCore = dataLineBigCore - 1;
        coreUsed = GetCeilInt(dataLine, dataLineBigCore);
        bigCoreNum = dataLine - dataLineSmallCore * coreUsed;
    }
    uint64_t ubTailNum;
    uint64_t taskNum, taskEachLine, taskLastLine;
    if (tail % paramsNumPerBlock == 0 && tail < MAX_DEAL_NUM / gradDsize) {
        tilingMode = 1;
        ubTailNum =  tail;
        auto availSize = ubSize - ubTailNum * BUFFER_NUM * gradDsize;
        taskEachLine = std::min(dataLineBigCore, availSize / gradDsize / (tail + 1));
        taskNum = GetCeilInt(dataLineBigCore, taskEachLine);
        taskLastLine = dataLineBigCore - taskEachLine * (taskNum - 1);

        indexUbSize = CeilValue(taskEachLine, paramsNumPerBlock);
        gradInUbSize = tail * taskEachLine;
        gradOutUbSize = ubTailNum;
    } else {
        tilingMode = 0;
        auto availIndicesSize = ubSize - std::min(CeilValue(tail, paramsNumPerBlock), MAX_DEAL_NUM) * BUFFER_NUM * gradDsize;
        ComputeTaskNumForeLine(availIndicesSize / indexDsize, dataLineBigCore, &taskNum, &taskEachLine, &taskLastLine);

        indexUbSize = std::min(availIndicesSize / indexDsize, dataLineBigCore);
        ubTailNum = (ubSize - indexUbSize * indexDsize) / BUFFER_NUM / BLOCK_BYTES * indexNumPerBlock;
        ubTailNum = std::min(ubTailNum, CeilValue(tail, paramsNumPerBlock));
        gradInUbSize = 0;
        gradOutUbSize = ubTailNum * BUFFER_NUM;
    }
    TilingData.set_taskNum(taskNum);
    TilingData.set_taskEachLine(taskEachLine);
    TilingData.set_taskLastLine(taskLastLine);
    TilingData.set_ubTailNum(ubTailNum);
    TilingData.set_bacthSmallCore(dataLineSmallCore);
    TilingData.set_gradInUbSize(gradInUbSize);
    TilingData.set_gradOutUbSize(gradOutUbSize);
}

ge::graphStatus ScatterMeanGradTiling::Init()
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t totalUbSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, totalUbSize);
    ubSize = totalUbSize - RESERVE_SAPCE;

    if (context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr || context->GetInputShape(2) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto gradOutShape = context->GetInputShape(0)->GetStorageShape();
    auto indexShape = context->GetInputShape(1)->GetStorageShape();
    auto countShape = context->GetInputShape(2)->GetStorageShape();
    auto gradInShape = context->GetOutputShape(0)->GetStorageShape();
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t* axisPtr = attrs->GetAttrPointer<int64_t>(0);
    int32_t axis = static_cast<int32_t>(*axisPtr);

    // check inputs shape
    int32_t gradDims = gradOutShape.GetDimNum();
    int32_t indexDims = indexShape.GetDimNum();
    indexDims = indexDims == 0 ? 1 : indexDims;
    axis = (axis + indexDims) % indexDims;
    dim = axis;

    for (int32_t i = 0; i < axis; i++) {
        paramsPre *= gradInShape.GetDim(i);
    }
    dimRange = gradInShape.GetDim(axis);
    dimRangeOut = gradOutShape.GetDim(axis);
    if (dimRange == 0 || dimRangeOut == 0 || paramsPre == 0) {
        return ge::GRAPH_FAILED;
    }
    for (int32_t i = axis + 1; i < gradDims; i++) {
        paramsPro *= gradInShape.GetDim(i);
    }
    for (int32_t i = indexDims; i < gradDims; i++) {
        tail *= gradInShape.GetDim(i);
    }
    gradInNum = paramsPre * dimRange * paramsPro;
    for (int32_t i = 0; i < indexDims; i++) {
        indexNum *= indexShape.GetDim(i);
    }
    gradOutNum = paramsPre * dimRangeOut * paramsPro;
    countNum = countShape.GetShapeSize();

    if (context->GetInputDesc(0) == nullptr || context->GetInputDesc(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto gradDtype = context->GetInputDesc(0)->GetDataType();
    gradDsize = sizeof(gradDtype);
    paramsNumPerBlock = BLOCK_BYTES / gradDsize;
    auto indexDtype = context->GetInputDesc(1)->GetDataType();
    indexDsize = sizeof(indexDtype);

    if (tail == 1) {
        SetModeNoTail(gradDims, indexDims, coreNum);
    } else {
        SetModeLine(gradDims, indexDims, coreNum);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterMeanGradTiling::RunKernelTiling()
{
    context->SetBlockDim(coreUsed);
    TilingData.set_tilingMode(tilingMode);
    TilingData.set_dimRange(dimRange);
    TilingData.set_dimRangeOut(dimRangeOut);
    TilingData.set_paramsPro(paramsPro);

    TilingData.set_gradInUbSize(gradInUbSize);
    TilingData.set_indexUbSize(indexUbSize);
    TilingData.set_gradOutUbSize(gradOutUbSize);
    TilingData.set_indexSumUbSize(indexSumUbSize);
    
    TilingData.set_gradInNum(gradInNum);
    TilingData.set_indexNum(indexNum);
    TilingData.set_gradOutNum(gradOutNum);
    TilingData.set_countNum(countNum);
    TilingData.set_bigCoreNum(bigCoreNum);
    TilingData.set_tail(tail);

    size_t sysWorkspaceSize = WORKSPACE_16MBYTE_SIZE;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    TilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc4ScatterMeanGrad(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ScatterMeanGradTiling tilingObject(context);
    tilingObject.Init();

    return tilingObject.RunKernelTiling();
}
}


namespace ge {
static ge::graphStatus InferShape4ScatterMeanGrad(gert::InferShapeContext* context)
{
    const gert::Shape *gradOutShape = context->GetInputShape(0);
    if (gradOutShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *indexShape = context->GetInputShape(1);
    if (indexShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int32_t axis = *(attrs->GetAttrPointer<int32_t>(0));
    int32_t gradDims = gradOutShape->GetDimNum();
    gradDims = gradDims == 0 ? 1 : gradDims;
    axis = (axis + gradDims) % gradDims;

    // check inputs shape
    gert::Shape *gradInShape = context->GetOutputShape(0);
    if (gradInShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *gradInShape = *gradOutShape;
    gradInShape->SetDim(axis, indexShape->GetDim(axis));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4ScatterMeanGrad(gert::InferDataTypeContext *context)
{
    const ge::DataType grad_out_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, grad_out_dtype);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class ScatterMeanGrad : public OpDef {
public:
    explicit ScatterMeanGrad(const char* name) : OpDef(name)
    {
        this->Input("grad_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("count")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_in")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("dim").Int();
        this->SetInferShape(ge::InferShape4ScatterMeanGrad)
             .SetInferDataType(ge::InferDtype4ScatterMeanGrad);
        this->AICore()
            .SetTiling(optiling::TilingFunc4ScatterMeanGrad);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(ScatterMeanGrad);
}