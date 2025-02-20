/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#include "scatter_add_tiling.h"
#include "common.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

using namespace std;

namespace optiling {
const uint64_t BLOCK_SIZE = 32;
const uint64_t MAX_COPY_PAD =  4095;
const uint64_t MAX_DEAL_NUM =  2048;
const uint64_t INDICES_ONCE_DATANUM = 2048;
const uint64_t TILING_MODE_NO_TAIL_MULTIHEAD = 3;
const uint64_t TILING_MODE_NO_TAIL = 2;
const uint64_t TILING_MODE_NORMAL = 1;
constexpr uint64_t ONTAIL_INDICES_UB_NUM = 2;
constexpr uint64_t BUFFER_NUM_MAX = 8;


class ScatterAddTiling {
public:
    ScatterAddTiling() {}
    ge::graphStatus Init(gert::TilingContext* context);
    ge::graphStatus GetKernelTiling(gert::TilingContext* context);
    ge::graphStatus SetKernelTiling(gert::TilingContext* context);
private:
    void ComputeTask(uint64_t ubOutNum, uint64_t outLineEachCore, uint32_t &taskNum, uint64_t &taskEachLine, uint64_t &taskLastLine);
    ge::graphStatus getUBNumMulitHead(gert::TilingContext* context, uint64_t indicesEachHead,
        uint64_t outNumEachHead, uint64_t &ubOutNum, uint64_t &indicesDealNum, uint64_t &headNum);
    ge::graphStatus ScatterAddNoTailTilingFunc(gert::TilingContext* context);
    ge::graphStatus ScatterAddNormalTilingFunc(gert::TilingContext* context);

    ScatterAddTilingData TilingData;

    uint64_t usedCoreNum;
    uint64_t dim;
    uint64_t head;
    uint64_t tail;
    uint64_t ubSize;
    uint64_t coreNum;
    uint64_t outNum;
    uint64_t indicesNum;
    uint64_t srcNum;
    uint64_t outDimShape;
    uint64_t dimShape;
    uint64_t dataDsize;
    uint64_t indicesDsize;
    uint64_t dataEachBlock;
    uint64_t bigCoreNum;

    uint32_t taskNum, taskNumLast;
    uint32_t tilingMode = 0;
    uint64_t taskEachLine, taskLastLine;
};

void ScatterAddTiling::ComputeTask(uint64_t ubOutNum, uint64_t outLineEachCore, uint32_t &taskNum, uint64_t &taskEachLineA, uint64_t &taskLastLineA)
{
    // 按核数，ub和数据量分任务
    if (outLineEachCore <= ubOutNum) {
        taskNum = 1;
        taskEachLineA = outLineEachCore;
        taskLastLineA = outLineEachCore;
    } else {
        uint64_t taskNumTemp = DivCeil(outLineEachCore, ubOutNum);
        taskNum = taskNumTemp;
        taskEachLineA = ubOutNum;
        taskLastLineA = outLineEachCore - ubOutNum * (taskNumTemp - 1);
    }
}

ge::graphStatus ScatterAddTiling::getUBNumMulitHead(gert::TilingContext* context, uint64_t indicesEachHead, uint64_t outNumEachHead, uint64_t &ubOutNum, uint64_t &indicesDealNum, uint64_t &headNum)
{
    uint64_t tempHeadNum = ubSize / BLOCK_SIZE * dataEachBlock / (ONTAIL_INDICES_UB_NUM * indicesEachHead + outNumEachHead);
    headNum = tempHeadNum;

    if (tempHeadNum == 0) {
        uint64_t ubIndicesNumTemp = std::min(INDICES_ONCE_DATANUM, indicesEachHead);
        uint64_t ubAvailableBytes = ubSize - ubIndicesNumTemp * ONTAIL_INDICES_UB_NUM * indicesDsize;
        ubOutNum = ubAvailableBytes / BLOCK_SIZE * dataEachBlock;
        indicesDealNum = ubIndicesNumTemp;
    } else {
        // new
        ubOutNum  = tempHeadNum * outNumEachHead;
        indicesDealNum = tempHeadNum * indicesEachHead;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTiling::ScatterAddNoTailTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t indicesDealNum;
    uint64_t ubOutNum;
    uint64_t indicesEachHead = indicesNum / head;
    uint64_t outNumEachHead = outNum / head;

    uint64_t lineSmallCore = 1;
    uint64_t lineBigCore = 1;
    uint64_t outLineEachCore;
    uint64_t outLineLastBigCore;
    bigCoreNum = coreNum;

    usedCoreNum = coreNum;
    uint64_t coreEachHead = 1;
    uint64_t taskEachLineLast;
    uint64_t taskLastLineLast;
    
    if (head > coreNum) {
        uint64_t headNum;
        getUBNumMulitHead(context, indicesEachHead, outNumEachHead, ubOutNum, indicesDealNum, headNum);
        lineSmallCore = head / coreNum;
        lineBigCore = lineSmallCore + 1;
        bigCoreNum = head - lineSmallCore * coreNum;

        uint64_t headNumEachTask = std::min(headNum, lineBigCore); // 能放下的最多的head数和实际需要处理的head数min
        if (headNumEachTask == 0) {
            context->SetTilingKey(TILING_MODE_NO_TAIL);
            outLineEachCore = outDimShape;
            outLineLastBigCore = outDimShape;
            ComputeTask(ubOutNum, outLineEachCore, taskNum, taskEachLine, taskLastLine);
            ComputeTask(ubOutNum, outLineLastBigCore, taskNumLast, taskEachLineLast, taskLastLineLast);
        } else {
            context->SetTilingKey(TILING_MODE_NO_TAIL_MULTIHEAD);
            ubOutNum = headNumEachTask * outNumEachHead;
            indicesDealNum = headNumEachTask * indicesEachHead;
            taskNum = DivCeil(lineBigCore, headNumEachTask);
            uint64_t headNumBigLast = lineBigCore - (taskNum - 1) * headNumEachTask;
            taskEachLine = headNumEachTask * outDimShape;
            taskLastLine = headNumBigLast * outDimShape;

            taskNumLast = DivCeil(lineSmallCore, headNumEachTask);
            uint64_t headNumSmallLast = lineSmallCore - (taskNumLast - 1) * headNumEachTask;
            taskEachLineLast = taskEachLine;
            taskLastLineLast = headNumSmallLast * outDimShape;
            TilingData.set_headNumEachTask(headNumEachTask);
            TilingData.set_headNumBigLast(headNumBigLast);
            TilingData.set_headNumSmallLast(headNumSmallLast);
        }
    } else {
        // head较小，一个head分在多个核
        tilingMode = 1;
        uint64_t ubIndicesNumTemp = std::min(INDICES_ONCE_DATANUM, indicesEachHead);
        uint64_t ubAvailableBytes = ubSize - ubIndicesNumTemp * ONTAIL_INDICES_UB_NUM * indicesDsize;
        ubOutNum = ubAvailableBytes / BLOCK_SIZE * dataEachBlock;
        indicesDealNum = ubIndicesNumTemp;

        context->SetTilingKey(TILING_MODE_NO_TAIL);
        coreEachHead = std::min(coreNum / head, outDimShape);
        bigCoreNum = usedCoreNum;
        outLineEachCore = DivCeil(outDimShape, coreEachHead);
        coreEachHead = DivCeil(outDimShape, outLineEachCore);
        usedCoreNum = head * coreEachHead;
        outLineLastBigCore = outDimShape - outLineEachCore * (coreEachHead - 1);
        ComputeTask(ubOutNum, outLineEachCore, taskNum, taskEachLine, taskLastLine);
        ComputeTask(ubOutNum, outLineLastBigCore, taskNumLast, taskEachLineLast, taskLastLineLast);
    }

    TilingData.set_head(head);
    TilingData.set_lineSmallCore(lineSmallCore);
    TilingData.set_lineBigCore(lineBigCore);
    TilingData.set_outLineEachCore(outLineEachCore);
    TilingData.set_coreEachHead(coreEachHead);
    TilingData.set_taskNumLast(taskNumLast);
    TilingData.set_taskEachLineLast(taskEachLineLast);
    TilingData.set_taskLastLineLast(taskLastLineLast);
    TilingData.set_indicesDealNum(indicesDealNum);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTiling::ScatterAddNormalTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t lineSmallCore = 1;
    uint64_t lineBigCore = 1;
    usedCoreNum = coreNum;

    uint64_t dataLine = dimShape * head;

    lineBigCore = DivCeil(dataLine, coreNum);
    lineSmallCore = lineBigCore - 1;
    usedCoreNum = DivCeil(dataLine, lineBigCore);
    bigCoreNum = dataLine - lineSmallCore * usedCoreNum;

    uint64_t indicesDealNum;
    uint64_t ubTailNum;
    uint64_t ubSrcNum;
    uint32_t dbTimes;

    if (tail <= MAX_DEAL_NUM) {
        tilingMode = 1;
        dbTimes = 1;
        ubTailNum = CeilAlign(tail, dataEachBlock);
        taskEachLine = std::min(lineBigCore, ubSize / dataDsize / (ubTailNum + 1));
        taskEachLine = std::min(taskEachLine, MAX_COPY_PAD);
        taskNum = DivCeil(lineBigCore, taskEachLine);
        taskLastLine = lineBigCore - taskEachLine * (taskNum - 1);

        indicesDealNum = CeilAlign(taskEachLine, dataEachBlock);
        ubSrcNum = ubTailNum * taskEachLine;
    } else {
        tilingMode = 0;
        uint64_t totalTailUb = ubSize - taskEachLine * indicesDsize;
        dbTimes = std::min(totalTailUb / dataDsize / std::min(tail, MAX_DEAL_NUM), BUFFER_NUM_MAX);

        uint64_t availIndicesSize = ubSize - std::min(CeilAlign(tail, dataEachBlock), MAX_DEAL_NUM) * dataDsize * dbTimes;
        ComputeTask(availIndicesSize / 4, lineBigCore, taskNum, taskEachLine, taskLastLine);
        indicesDealNum = std::min(availIndicesSize / BLOCK_SIZE * BLOCK_SIZE / indicesDsize, lineBigCore);
        if (dbTimes == 0) {
            ubTailNum = (ubSize - indicesDealNum * indicesDsize) / BLOCK_SIZE * dataEachBlock;
        } else {
            ubTailNum = (ubSize - indicesDealNum * indicesDsize) / dbTimes / BLOCK_SIZE * dataEachBlock;
        }
        ubTailNum = std::min(ubTailNum, CeilAlign(tail, dataEachBlock));
        ubSrcNum = ubTailNum * dbTimes;
    }

    TilingData.set_lineSmallCore(lineSmallCore);
    TilingData.set_indicesDealNum(indicesDealNum);
    TilingData.set_ubTailNum(ubTailNum);
    TilingData.set_ubSrcNum(ubSrcNum);
    TilingData.set_dbTimes(dbTimes);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTiling::Init(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize =  ubSize - 8 * 1024;

    if (context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr || context->GetInputShape(2) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto srcShape = context->GetInputShape(0)->GetStorageShape();
    auto indicesShape = context->GetInputShape(1)->GetStorageShape();
    auto varShape = context->GetInputShape(2)->GetStorageShape();

    outNum = varShape.GetShapeSize();
    indicesNum = indicesShape.GetShapeSize();
    srcNum = srcShape.GetShapeSize();

    uint64_t srcDim = srcShape.GetDimNum();
    uint64_t indicesDim = indicesShape.GetDimNum();
    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    dim = *(attrsPtr->GetAttrPointer<int>(0));

    outDimShape = varShape.GetDim(dim);
    dimShape = srcShape.GetDim(dim);
    if (dim < 0 || dim >= srcDim || dim >= indicesDim) {
        return ge::GRAPH_FAILED;
    }

    // size of dataType
    head = 1;
    for (uint64_t i = 0; i < dim; i++) {
        head *= srcShape.GetDim(i);
    }
    tail = 1;
    for (uint64_t i = indicesDim; i < srcDim; i++) {
        tail *= srcShape.GetDim(i);
    }

    if (context->GetInputDesc(0) == nullptr || context->GetInputDesc(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dataDtype = context->GetInputDesc(0)->GetDataType();
    auto indicesDtype = context->GetInputDesc(1)->GetDataType();
    dataDsize = kDataSizeMap[dataDtype];       // now only support float32
    indicesDsize = kDataSizeMap[indicesDtype]; // now only support int32
    dataEachBlock = BLOCK_SIZE / dataDsize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTiling::GetKernelTiling(gert::TilingContext* context)
{
    if (tail == 1) {
        return ScatterAddNoTailTilingFunc(context);
    } else {
        context->SetTilingKey(TILING_MODE_NORMAL);
        return ScatterAddNormalTilingFunc(context);
    }
}

ge::graphStatus ScatterAddTiling::SetKernelTiling(gert::TilingContext* context)
{
    context->SetBlockDim(usedCoreNum);
    TilingData.set_bigCoreNum(bigCoreNum);
    TilingData.set_outNum(outNum);
    TilingData.set_indicesNum(indicesNum);
    TilingData.set_srcNum(srcNum);
    TilingData.set_tail(tail);
    TilingData.set_taskNum(taskNum);
    TilingData.set_taskEachLine(taskEachLine);
    TilingData.set_taskLastLine(taskLastLine);
    TilingData.set_outDimSize(outDimShape);
    TilingData.set_dimSize(dimShape);
    TilingData.set_tilingMode(tilingMode);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    TilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTilingFunc(gert::TilingContext* context)
{
    ScatterAddTiling tilingObject;

    if (tilingObject.Init(context) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.GetKernelTiling(context) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    return tilingObject.SetKernelTiling(context);
}
}


namespace ge {
static ge::graphStatus ScatterAddInferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (x1_shape == nullptr || y_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus ScatterAddInferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType src_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, src_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ScatterAddV2 : public OpDef {
public:
    explicit ScatterAddV2(const char* name) : OpDef(name)
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
        this->Attr("dim").Int();

        this->SetInferShape(ge::ScatterAddInferShape)
              .SetInferDataType(ge::ScatterAddInferDataType);

        this->AICore()
            .SetTiling(optiling::ScatterAddTilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(ScatterAddV2);
}