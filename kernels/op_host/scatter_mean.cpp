/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "scatter_mean.h"
#include "common.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

using namespace std;

namespace optiling {
const uint64_t BLOCK_SIZE = 32;
const uint64_t MAX_OUT_LINE =  16000;
const uint64_t MAX_DEAL_NUM =  2048;
const uint64_t INDICES_ONCE_DATANUM = 2048;
const uint64_t TILING_MODE_NO_TAIL_MULTIHEAD = 3;
const uint64_t TILING_MODE_NO_TAIL = 2;
const uint64_t TILING_MODE_NORMAL = 1;
const uint64_t LEAST_LINE_EACH_TASK = 4;

static uint64_t GetCeilInt(uint64_t value1, uint64_t value2)
{
    if (value2 == 0) {
        return value1;
    }
    return (value1 + value2 - 1) / value2;
}

static void ComputeTaskForBatch(uint64_t ubOutNum, uint64_t outLineEachBacth, uint64_t *taskNum, uint64_t *taskEachLine, uint64_t *taskLastLine)
{
    if (outLineEachBacth <= ubOutNum) {
        *taskNum = 1;
        *taskEachLine = outLineEachBacth;
        *taskLastLine = outLineEachBacth;
    } else {
        uint64_t taskNumTemp = GetCeilInt(outLineEachBacth, ubOutNum);
        *taskNum = taskNumTemp;
        *taskEachLine = ubOutNum;
        *taskLastLine = outLineEachBacth - ubOutNum * (taskNumTemp - 1);
    }
}

static ge::graphStatus ScatterMeanGetUBNum(gert::TilingContext* context, uint64_t indicesNumEachHead, uint64_t *ubOutNum, uint64_t *ubIndicesNum)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t UB_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, UB_size);

    auto dataDtype = context->GetInputDesc(0)->GetDataType();
    auto indicesDtype = context->GetInputDesc(1)->GetDataType();
    uint64_t bytesData = kDataSizeMap[dataDtype];       // now only support float32
    uint64_t bytesIndices = kDataSizeMap[indicesDtype]; // now only support int32
    auto dataEachBlock = BLOCK_SIZE / bytesData;

    uint64_t ubIndicesNumTemp = std::min(INDICES_ONCE_DATANUM, indicesNumEachHead);
    uint64_t ubAvailableBytes = UB_size - ubIndicesNumTemp * 2 * bytesIndices - 8 * 1024;
    *ubOutNum = ubAvailableBytes / 2 / BLOCK_SIZE * dataEachBlock;
    *ubIndicesNum = ubIndicesNumTemp;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ScatterMeanGetUBNumMulitHead(gert::TilingContext* context, uint64_t indicesNumEachHead, uint64_t outNumEachHead, uint64_t *ubOutNum, uint64_t *ubIndicesNum, uint64_t * headNum)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t UB_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, UB_size);

    if (context->GetInputDesc(0) == nullptr || context->GetInputDesc(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dataDtype = context->GetInputDesc(0)->GetDataType();
    auto indicesDtype = context->GetInputDesc(1)->GetDataType();
    uint64_t bytesData = kDataSizeMap[dataDtype];       // now only support float32
    uint64_t bytesIndices = kDataSizeMap[indicesDtype]; // now only support int32
    auto dataEachBlock = BLOCK_SIZE / bytesData;

    UB_size =  UB_size - 8 * 1024;
    uint64_t tempHeadNum = UB_size / BLOCK_SIZE * dataEachBlock / (indicesNumEachHead + outNumEachHead) / 2;
    *headNum = tempHeadNum;

    if (tempHeadNum == 0) {
        uint64_t ubIndicesNumTemp = std::min(INDICES_ONCE_DATANUM, indicesNumEachHead);
        uint64_t ubAvailableBytes = UB_size - ubIndicesNumTemp * 2 * bytesIndices;
        *ubOutNum = ubAvailableBytes / 2 / BLOCK_SIZE * dataEachBlock;
        *ubIndicesNum = ubIndicesNumTemp;
    } else {
        // new
        *ubOutNum  = tempHeadNum * outNumEachHead;
        *ubIndicesNum = tempHeadNum * indicesNumEachHead;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ScatterMeanNoTailTilingFunc(gert::TilingContext* context)
{
    ScatterMeanTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static uint64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t outTailNum = 1;
    if (context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr || context->GetInputShape(2) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto srcShape = context->GetInputShape(0)->GetStorageShape();
    auto indicesShape = context->GetInputShape(1)->GetStorageShape();
    auto varShape = context->GetInputShape(2)->GetStorageShape();

    uint64_t outNum = varShape.GetShapeSize();
    uint64_t indicesNum = indicesShape.GetShapeSize();
    uint64_t srcNum = srcShape.GetShapeSize();

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint64_t dim = *(attrsPtr->GetAttrPointer<int>(0));

    uint64_t head = 1;
    for (uint64_t i = 0; i < dim; i++) {
        head *= srcShape.GetDim(i);
    }

    if (outNum == 0 || indicesNum == 0 || srcNum == 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ubIndicesNum;
    uint64_t ubOutNum;
    uint64_t indicesNumEachHead = indicesNum / head;
    uint64_t outNumEachHead = outNum / head;
    uint64_t headNum;

    uint64_t bacthSmallCore = 1;
    uint64_t bacthBigCore = 1;
    uint64_t outLineEachBacth;
    uint64_t taskEachBacth = 1;
    uint64_t bigCoreNum = coreNum;
    uint64_t outLineLastBigBatch;

    uint64_t usedCoreNum = coreNum;
    uint64_t coreEachHead = 1;
    uint64_t out_dim_shape = varShape.GetDim(dim);

    uint64_t taskNum, taskEachLine, taskLastLine;
    uint64_t taskNumLast, taskEachLineLast, taskLastLineLast;
    
    if (head > coreNum) {
        context->SetTilingKey(TILING_MODE_NO_TAIL_MULTIHEAD);
        ScatterMeanGetUBNumMulitHead(context, indicesNumEachHead, outNumEachHead, &ubOutNum, &ubIndicesNum, &headNum);
        bacthSmallCore = head / coreNum;
        bacthBigCore = bacthSmallCore + 1;
        bigCoreNum = head - bacthSmallCore * coreNum;

        uint64_t headNumEachTask = std::min(headNum, bacthBigCore);
        if (headNumEachTask == 0) {
            outLineEachBacth = out_dim_shape;
            outLineLastBigBatch = out_dim_shape;
            ComputeTaskForBatch(ubOutNum, outLineEachBacth, &taskNum, &taskEachLine, &taskLastLine);
            ComputeTaskForBatch(ubOutNum, outLineLastBigBatch, &taskNumLast, &taskEachLineLast, &taskLastLineLast);
        } else {
            ubOutNum = headNumEachTask * outNumEachHead;
            ubIndicesNum = headNumEachTask * indicesNumEachHead;
            taskNum = GetCeilInt(bacthBigCore, headNumEachTask);
            uint64_t headNumBigLast = bacthBigCore - (taskNum - 1) * headNumEachTask;
            taskEachLine = headNumEachTask * out_dim_shape;
            taskLastLine = headNumBigLast * out_dim_shape;

            taskNumLast = GetCeilInt(bacthSmallCore, headNumEachTask);
            uint64_t headNumSmallLast = bacthSmallCore - (taskNumLast - 1) * headNumEachTask;
            taskEachLineLast = taskEachLine;
            taskLastLineLast = headNumSmallLast * out_dim_shape;
            tiling.set_headNumEachTask(headNumEachTask);
            tiling.set_headNumBigLast(headNumBigLast);
            tiling.set_headNumSmallLast(headNumSmallLast);
        }
    } else {
        ScatterMeanGetUBNum(context, indicesNumEachHead, &ubOutNum, &ubIndicesNum);
        context->SetTilingKey(TILING_MODE_NO_TAIL);
        coreEachHead = std::min(coreNum / head, out_dim_shape);
        bigCoreNum = usedCoreNum;
        outLineEachBacth = GetCeilInt(out_dim_shape, coreEachHead);
        coreEachHead = GetCeilInt(out_dim_shape, outLineEachBacth);
        usedCoreNum = head * coreEachHead;
        outLineLastBigBatch = out_dim_shape - outLineEachBacth * (coreEachHead - 1);
        ComputeTaskForBatch(ubOutNum, outLineEachBacth, &taskNum, &taskEachLine, &taskLastLine);
        ComputeTaskForBatch(ubOutNum, outLineLastBigBatch, &taskNumLast, &taskEachLineLast, &taskLastLineLast);
    }

    uint64_t indicesLoop = indicesNumEachHead / ubIndicesNum;
    uint64_t indicesLastNum = indicesNumEachHead % ubIndicesNum;

    context->SetBlockDim(usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_outNum(outNum);
    tiling.set_indicesNum(indicesNum);
    tiling.set_srcNum(srcNum);
    tiling.set_bigCoreNum(bigCoreNum);
    tiling.set_head(head);
    tiling.set_bacthSmallCore(bacthSmallCore);
    tiling.set_bacthBigCore(bacthBigCore);
    tiling.set_taskNum(taskNum);
    tiling.set_taskEachLine(taskEachLine);
    tiling.set_taskLastLine(taskLastLine);
    tiling.set_outLineEachBacth(outLineEachBacth);
    tiling.set_coreEachHead(coreEachHead);
    tiling.set_taskNumLast(taskNumLast);
    tiling.set_taskEachLineLast(taskEachLineLast);
    tiling.set_taskLastLineLast(taskLastLineLast);
    tiling.set_indicesLoop(indicesLoop);
    tiling.set_indicesLastNum(indicesLastNum);
    tiling.set_ubIndicesNum(ubIndicesNum);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static uint64_t GetAvailableDimNum(gert::TilingContext* context)
{
    auto indicesShape = context->GetInputShape(1)->GetStorageShape();
    uint64_t indicesDim = indicesShape.GetDimNum();
    uint64_t lastIndicesDim = 0;
    for (uint64_t i = indicesDim - 1; i >= 0; i--) {
        if (indicesShape.GetDim(i) == 1) {
            lastIndicesDim++;
        } else {
            break;
        }
    }
    return indicesDim - lastIndicesDim;
}

static ge::graphStatus ScatterMeanNormalTilingFunc(gert::TilingContext* context)
{
    ScatterMeanTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static uint64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t UB_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, UB_size);

    uint64_t outTailNum = 1;
    if (context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr || context->GetInputShape(2) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto srcShape = context->GetInputShape(0)->GetStorageShape();
    auto indicesShape = context->GetInputShape(1)->GetStorageShape();
    auto varShape = context->GetInputShape(2)->GetStorageShape();

    uint64_t srcDim = srcShape.GetDimNum();
    uint64_t indicesDim = indicesShape.GetDimNum();
    uint64_t outNum = varShape.GetShapeSize();
    uint64_t indicesNum = indicesShape.GetShapeSize();
    uint64_t srcNum = srcShape.GetShapeSize();

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint64_t dim = *(attrsPtr->GetAttrPointer<int>(0));

    uint64_t head = 1;
    for (uint64_t i = 0; i < dim; i++) {
        head *= srcShape.GetDim(i);
    }
    uint64_t body = 1;
    for (uint64_t i = dim + 1; i < indicesDim; i++) {
        body *= srcShape.GetDim(i);
    }
    uint64_t tail = 1;
    uint64_t availIindicesDim = GetAvailableDimNum(context);
    for (uint64_t i = availIindicesDim; i < srcDim; i++) {
        tail *= srcShape.GetDim(i);
    }

    uint64_t dimShape = srcShape.GetDim(dim);
    uint64_t bigCoreNum = coreNum;
    uint64_t bacthSmallCore = 1;
    uint64_t bacthBigCore = 1;
    uint64_t usedCoreNum = coreNum;

    uint64_t dataLine = dimShape * head * body;
    if (dataLine <= 2 * LEAST_LINE_EACH_TASK) {
        usedCoreNum = 1;
        bacthBigCore = dataLine;
        bigCoreNum = 1;
    } else {
        bacthBigCore = std::max(GetCeilInt(dataLine, coreNum), LEAST_LINE_EACH_TASK);
        bacthSmallCore = bacthBigCore - 1;
        usedCoreNum = GetCeilInt(dataLine, bacthBigCore);
        bigCoreNum = dataLine - bacthSmallCore * usedCoreNum;
    }

    if (context->GetInputDesc(0) == nullptr || context->GetInputDesc(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dataDtype = context->GetInputDesc(0)->GetDataType();
    auto indicesDtype = context->GetInputDesc(1)->GetDataType();
    uint64_t bytesData = kDataSizeMap[dataDtype];       // now only support float32
    uint64_t bytesIndices = kDataSizeMap[indicesDtype]; // now only support int32
    auto dataEachBlock = BLOCK_SIZE / bytesData;

    uint64_t taskNum, taskEachLine, taskLastLine;
    uint64_t taskNumLast, taskEachLineLast, taskLastLineLast;
    ComputeTaskForBatch(MAX_OUT_LINE, bacthBigCore, &taskNum, &taskEachLine, &taskLastLine);
    ComputeTaskForBatch(MAX_OUT_LINE, bacthSmallCore, &taskNumLast, &taskEachLineLast, &taskLastLineLast);

    uint64_t ubIndicesNum;
    UB_size = UB_size - 8 * 1024;
    ubIndicesNum = UB_size - std::min(tail, MAX_DEAL_NUM) * bytesData;
    ubIndicesNum = std::min(ubIndicesNum / BLOCK_SIZE * BLOCK_SIZE / bytesIndices, bacthBigCore);

    uint64_t ubTailNum = (UB_size - ubIndicesNum * bytesIndices) / BLOCK_SIZE * dataEachBlock;
    ubTailNum = std::min(ubTailNum, GetCeilInt(tail, dataEachBlock) * dataEachBlock);

    uint64_t outDimSize = varShape.GetDim(dim);

    context->SetBlockDim(usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_outNum(outNum);
    tiling.set_indicesNum(indicesNum);
    tiling.set_srcNum(srcNum);
    tiling.set_bigCoreNum(bigCoreNum);
    tiling.set_tail(tail);
    tiling.set_body(body);
    tiling.set_bacthSmallCore(bacthSmallCore);
    tiling.set_taskNum(taskNum);
    tiling.set_taskEachLine(taskEachLine);
    tiling.set_taskLastLine(taskLastLine);
    tiling.set_taskEachLineLast(taskEachLineLast);
    tiling.set_taskLastLineLast(taskLastLineLast);
    tiling.set_taskNumLast(taskNumLast);
    tiling.set_ubIndicesNum(ubIndicesNum);
    tiling.set_outDimSize(outDimSize);
    tiling.set_dimSize(dimShape);
    tiling.set_ubTailNum(ubTailNum);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ScatterMeanTilingFunc(gert::TilingContext* context)
{
    ScatterMeanTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    if (context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto srcShape = context->GetInputShape(0)->GetStorageShape();
    auto indicesShape = context->GetInputShape(1)->GetStorageShape();
    uint64_t srcDim = srcShape.GetDimNum();
    uint64_t indicesDim = indicesShape.GetDimNum();
    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint64_t dim = *(attrsPtr->GetAttrPointer<int>(0));

    uint64_t head = 1;
    if (dim < 0 || dim >= srcDim || dim >= indicesDim) {
        return ge::GRAPH_FAILED;
    }

    uint64_t availIindicesDim = GetAvailableDimNum(context);
    uint64_t tail = 1;
    for (uint64_t i = availIindicesDim; i < srcDim; i++) {
        tail *= srcShape.GetDim(i);
    }
    if (tail == 1) {
        ScatterMeanNoTailTilingFunc(context);
    } else {
        context->SetTilingKey(TILING_MODE_NORMAL);
        ScatterMeanNormalTilingFunc(context);
    }
    return ge::GRAPH_SUCCESS;
}

}


namespace ge {
static ge::graphStatus ScatterMeanInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (x1_shape == nullptr || y_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus ScatterMeanInferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType src_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, src_dtype);
    context->SetOutputDataType(1, src_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ScatterMean : public OpDef {
public:
    explicit ScatterMean(const char* name) : OpDef(name)
    {
        this->Input("src")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("count")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("dim").Int();

        this->SetInferShape(ge::ScatterMeanInferShape)
              .SetInferDataType(ge::ScatterMeanInferDataType);

        this->AICore()
            .SetTiling(optiling::ScatterMeanTilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(ScatterMean);
}

/***********scatterMeanDiv***********/
namespace optiling {
static ge::graphStatus ScatterMeanDivTilingFunc2(gert::TilingContext* context)
{
    ScatterMeanDivTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static uint64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t UB_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, UB_size);

    uint64_t outTailNum = 1;
    auto srcShape = context->GetInputShape(0)->GetStorageShape();
    auto countShape = context->GetInputShape(1)->GetStorageShape();
    auto outShape = context->GetOutputShape(0)->GetStorageShape();

    uint64_t srcDim = srcShape.GetDimNum();
    uint64_t outNum = outShape.GetShapeSize();
    uint64_t countNum = countShape.GetShapeSize();
    uint64_t srcNum = srcShape.GetShapeSize();
    if (outNum == 0 || countNum == 0 || srcNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t availIcountDim = GetAvailableDimNum(context);
    uint64_t tail = srcNum / countNum;

    uint64_t bigCoreNum = coreNum;
    uint64_t coreSmallLine = 1;
    uint64_t coreBigLine = 1;
    uint64_t usedCoreNum = coreNum;
    if (countNum <= 2 * LEAST_LINE_EACH_TASK) {
        usedCoreNum = 1;
        coreBigLine = countNum;
        bigCoreNum = 1;
    } else {
        coreBigLine = std::max(GetCeilInt(countNum, coreNum), LEAST_LINE_EACH_TASK);
        coreSmallLine = coreBigLine - 1;
        usedCoreNum = GetCeilInt(countNum, coreBigLine);
        bigCoreNum = countNum - coreSmallLine * usedCoreNum;
    }
    uint64_t taskNum, taskEachLine, taskLastLine;
    uint64_t taskNumSmall, taskEachLineSmall, taskLastLineSmall;
    ComputeTaskForBatch(MAX_OUT_LINE, coreBigLine, &taskNum, &taskEachLine, &taskLastLine);
    ComputeTaskForBatch(MAX_OUT_LINE, coreSmallLine, &taskNumSmall, &taskEachLineSmall, &taskLastLineSmall);

    auto dataDtype = context->GetInputDesc(0)->GetDataType();
    uint64_t bytesData = kDataSizeMap[dataDtype];       // now only support float32
    auto dataEachBlock = BLOCK_SIZE / bytesData;

    UB_size = UB_size - 8 * 1024;
    uint64_t ubCountNum = UB_size - std::min(tail, MAX_DEAL_NUM) * bytesData;
    ubCountNum = std::min(ubCountNum / BLOCK_SIZE * BLOCK_SIZE / bytesData, coreBigLine);

    uint64_t ubTailNum = (UB_size - ubCountNum * bytesData) / BLOCK_SIZE * BLOCK_SIZE / bytesData; // 32字节对齐
    ubTailNum = std::min(ubTailNum, GetCeilInt(tail, dataEachBlock) * dataEachBlock);

    context->SetBlockDim(usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_outNum(outNum);
    tiling.set_countNum(countNum);
    tiling.set_srcNum(srcNum);
    tiling.set_bigCoreNum(bigCoreNum);
    tiling.set_tail(tail);
    tiling.set_coreSmallLine(coreSmallLine);
    tiling.set_coreBigLine(coreBigLine);
    tiling.set_taskNum(taskNum);
    tiling.set_taskEachLine(taskEachLine);
    tiling.set_taskLastLine(taskLastLine);
    tiling.set_taskEachLineSmall(taskEachLineSmall);
    tiling.set_taskLastLineSmall(taskLastLineSmall);
    tiling.set_taskNumSmall(taskNumSmall);
    tiling.set_ubCountNum(ubCountNum);
    tiling.set_ubTailNum(ubTailNum);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ScatterMeanDivTilingFunc(gert::TilingContext* context)
{
    ScatterMeanDivTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto srcShape = context->GetInputShape(0)->GetStorageShape();
    uint64_t srcDim = srcShape.GetDimNum();
    uint64_t availIcountDim = GetAvailableDimNum(context);
    uint64_t tail = 1;
    for (uint64_t i = availIcountDim; i < srcDim; i++) {
        tail *= srcShape.GetDim(i);
    }
    ScatterMeanDivTilingFunc2(context);

    return ge::GRAPH_SUCCESS;
}

}


namespace ge {
static ge::graphStatus ScatterMeanDivInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (x1_shape == nullptr || y_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus ScatterMeanDivInferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType src_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, src_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ScatterMeanDiv : public OpDef {
public:
    explicit ScatterMeanDiv(const char* name) : OpDef(name)
    {
        this->Input("src")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("count")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::ScatterMeanDivInferShape)
             .SetInferDataType(ge::ScatterMeanDivInferDataType);

        this->AICore()
            .SetTiling(optiling::ScatterMeanDivTilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(ScatterMeanDiv);
}
