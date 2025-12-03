/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#include "scatter_add_tiling_v3.h"
#include "common.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

using namespace std;

namespace optiling {
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t PRESERVED_UB_SIZE = 8 * 1024;
constexpr uint64_t ENTIRE_OUT_LOAD_THRESHOLD = 120 * 1024;
constexpr uint64_t DIM_SIZE_THRESHOLD = 200;
constexpr uint64_t TAIL_LEN_THRESHOLD = 2048;
constexpr uint64_t INDICES_IN_BATCH_NUM = 4096;
constexpr uint64_t UB_SIZE_COEFF = 2;

constexpr uint64_t TILING_KEY_NO_TAIL_FULLY_LOAD = 0;
constexpr uint64_t TILING_KEY_NO_TAIL_MULTI_HEADS = 1;
constexpr uint64_t TILING_KEY_NO_TAIL_LARGE_HEAD = 2;
constexpr uint64_t TILING_KEY_NO_TAIL_FEW_HEADS = 3;
constexpr uint64_t TILINR_KEY_WITH_SMALL_TAIL = 4;
constexpr uint64_t TILINR_KEY_WITH_LARGE_TAIL = 5;
}


namespace optiling {
class ScatterAddTilingV3 {
public:
    ScatterAddTilingV3() : totalHead(1), tailLen(1) {};
    ge::graphStatus SetKernelTilingData(gert::TilingContext* context);

private:
    ge::graphStatus setBaseTilingData(gert::TilingContext* context);
    ge::graphStatus setNoTailFullyLoadTilingData(gert::TilingContext* context); // tail length == 1
    ge::graphStatus setNoTailInBatchTilingData(gert::TilingContext* context);   // tail length == 1
    ge::graphStatus setWithTailTilingData(gert::TilingContext* context);        // tail length > 1

    
private:
    ScatterAddTilingDataV3 tilingData;

    uint64_t totalHead;
    uint64_t tailLen;
    uint64_t dimSize;
    uint64_t srcDimSize;
    uint64_t ubSize;
    uint64_t totalSrcNum;
    uint64_t totalOutNum;
    uint64_t totalIndicesNum;
    uint64_t indicesNumPerHead;
    uint64_t outNumPerHead;
    uint64_t tilingKey;

    uint64_t dim;
    uint64_t coreNum;
    uint64_t srcDSize;
    uint64_t outDSize;
    uint64_t indicesDSize;
    uint64_t totalOutSize;
    uint64_t dataNumPerBlock;
};

ge::graphStatus ScatterAddTilingV3::SetKernelTilingData(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    if (setBaseTilingData(context) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (tailLen == 1) {
        if (totalOutSize <= ENTIRE_OUT_LOAD_THRESHOLD) {
            // case 1: OUT can be fully loaded into ub
            setNoTailFullyLoadTilingData(context);
        } else {
            // case 2: process input in batches
            setNoTailInBatchTilingData(context);
        }
    } else {
        setWithTailTilingData(context);
    }

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTilingV3::setBaseTilingData(gert::TilingContext* context)
{
    // step 1: calculate the available ub size
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize -= PRESERVED_UB_SIZE;

    coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    // step 2: count the number of input elements
    if (context->GetInputShape(0) == nullptr || 
        context->GetInputShape(1) == nullptr || 
        context->GetInputShape(2) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto srcShape = context->GetInputShape(0)->GetStorageShape();
    auto varShape = context->GetInputShape(2)->GetStorageShape();
    auto indiceShape = context->GetInputShape(1)->GetStorageShape();

    totalSrcNum = srcShape.GetShapeSize();
    totalOutNum = varShape.GetShapeSize();
    totalIndicesNum = indiceShape.GetShapeSize();

    // step 3: calculate the number of heads and the length of tail 
    uint64_t srcDimNum = srcShape.GetDimNum();
    uint64_t indicesDimNum = indiceShape.GetDimNum();

    if (context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    dim = *(context->GetAttrs()->GetAttrPointer<int>(0));
    dimSize = varShape.GetDim(dim);
    srcDimSize = srcShape.GetDim(dim);

    for (uint64_t i = 0; i < dim; i++) {
        totalHead *= srcShape.GetDim(i);
    }
    for(uint64_t i = dim+1; i < srcDimNum; i++) {
        tailLen *= srcShape.GetDim(i);
    }

    // step 4: get data size
    if (context->GetInputDesc(0) == nullptr || context->GetInputDesc(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    srcDSize = kDataSizeMap[context->GetInputDesc(0)->GetDataType()];
    outDSize = kDataSizeMap[context->GetInputDesc(2)->GetDataType()];
    indicesDSize = kDataSizeMap[context->GetInputDesc(1)->GetDataType()];
    totalOutSize = totalOutNum * outDSize;

    if (totalHead == 0) {
        return ge::GRAPH_FAILED;
    }
    outNumPerHead = totalOutNum / totalHead;
    indicesNumPerHead = totalIndicesNum / totalHead;
    dataNumPerBlock = BLOCK_SIZE / srcDSize;

    // step 5: set base data
    tilingData.set_totalHead(totalHead);
    tilingData.set_tailLen(tailLen);
    tilingData.set_dimSize(dimSize);
    tilingData.set_ubSize(ubSize);
    tilingData.set_srcDimSize(srcDimSize);
    tilingData.set_totalSrcNum(totalSrcNum);
    tilingData.set_totalOutNum(totalOutNum);
    tilingData.set_totalIndicesNum(totalIndicesNum);
    tilingData.set_outNumPerHead(outNumPerHead);
    tilingData.set_indicesNumPerHead(indicesNumPerHead);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTilingV3::setNoTailFullyLoadTilingData(gert::TilingContext* context)
{
    // ensuring each used core processes at least one indices element
    uint64_t blockDim = totalIndicesNum > coreNum ? coreNum : totalIndicesNum;
    // load entire OUT into ub and process INDICES in batches
    uint64_t ubSizeExcludingOut = ubSize - totalOutSize;
    uint64_t maxLoadableIndicesNum;

    // vec operations requires additional memory in this case
    if (dimSize <= DIM_SIZE_THRESHOLD) {
        maxLoadableIndicesNum = ubSizeExcludingOut / (sizeof(srcDSize) * UB_SIZE_COEFF + sizeof(indicesDSize));
    } else {
        maxLoadableIndicesNum = ubSizeExcludingOut / (sizeof(srcDSize) + sizeof(indicesDSize));
    }

    uint64_t indicesNumBigCore = DivCeil(totalIndicesNum, coreNum);
    uint64_t indicesNumSmallCore = totalIndicesNum / coreNum;
    uint64_t bigCoreNum = totalIndicesNum - coreNum * indicesNumSmallCore;

    uint64_t indicesMaxLoadableNumBigCore = min(maxLoadableIndicesNum, indicesNumBigCore);
    uint64_t indicesMaxLoadableNumSmallCore = min(maxLoadableIndicesNum, indicesNumSmallCore);

    context->SetBlockDim(blockDim);
    context->SetTilingKey(TILING_KEY_NO_TAIL_FULLY_LOAD);
    tilingData.set_indicesNumBigCore(indicesNumBigCore);
    tilingData.set_indicesNumSmallCore(indicesNumSmallCore);
    tilingData.set_bigCoreNum(bigCoreNum);
    tilingData.set_indicesMaxLoadableNumBigCore(indicesMaxLoadableNumBigCore);
    tilingData.set_indicesMaxLoadableNumSmallCore(indicesMaxLoadableNumSmallCore);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTilingV3::setNoTailInBatchTilingData(gert::TilingContext* context)
{
    uint64_t indicesNumPerBatch = min(INDICES_IN_BATCH_NUM, indicesNumPerHead);
    uint64_t remainUbSize = ubSize - indicesNumPerBatch * indicesDSize * 2;
    uint64_t maxOutNumPerBatch = remainUbSize / BLOCK_SIZE * dataNumPerBlock;

    // case 1: a head devided into several cores
    if (totalHead < coreNum) {
        uint64_t coreNumPerHead = min(coreNum / totalHead, dimSize);
        uint64_t outNumPerCore = DivCeil(dimSize, coreNumPerHead);
        coreNumPerHead = DivCeil(dimSize, outNumPerCore);
        uint64_t usedCoreNum = totalHead * coreNumPerHead;

        context->SetBlockDim(usedCoreNum);
        context->SetTilingKey(TILING_KEY_NO_TAIL_FEW_HEADS);
        tilingData.set_coreNumPerHead(coreNumPerHead);
        tilingData.set_outNumPerCore(outNumPerCore);
        tilingData.set_indicesNumPerBatch(indicesNumPerBatch);
        tilingData.set_maxOutNumPerBatch(maxOutNumPerBatch);
        return ge::GRAPH_SUCCESS;
    }

    uint64_t headNumBigCore = DivCeil(totalHead, coreNum);
    uint64_t headNumSmallCore = totalHead / coreNum;
    uint64_t bigCoreNum = totalHead - coreNum * headNumSmallCore;
    uint64_t headNumPerTask = ubSize / BLOCK_SIZE * dataNumPerBlock / (2 * indicesNumPerHead + outNumPerHead);
    headNumPerTask = min(headNumBigCore, headNumPerTask);

    // case 2: at least one head can be fully loaded into ub
    if (headNumPerTask > 0) {
        context->SetBlockDim(coreNum);
        context->SetTilingKey(TILING_KEY_NO_TAIL_MULTI_HEADS);
        tilingData.set_headNumBigCore(headNumBigCore);
        tilingData.set_headNumSmallCore(headNumSmallCore);
        tilingData.set_bigCoreNum(bigCoreNum);
        tilingData.set_headNumPerTask(headNumPerTask);
        return ge::GRAPH_SUCCESS;
    }

    // case 3: can not even load one complete head, process head in batches
    context->SetBlockDim(coreNum);
    context->SetTilingKey(TILING_KEY_NO_TAIL_LARGE_HEAD);
    tilingData.set_headNumBigCore(headNumBigCore);
    tilingData.set_headNumSmallCore(headNumSmallCore);
    tilingData.set_bigCoreNum(bigCoreNum);
    tilingData.set_indicesNumPerBatch(indicesNumPerBatch);
    tilingData.set_maxOutNumPerBatch(maxOutNumPerBatch);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTilingV3::setWithTailTilingData(gert::TilingContext* context)
{
    uint64_t totalSrcTail = totalHead * srcDimSize; 
    uint64_t srcTailBigCore = DivCeil(totalSrcTail, coreNum);
    uint64_t srcTailSmallCore = srcTailBigCore - 1;
    uint64_t usedCoreNum = DivCeil(totalSrcTail, srcTailBigCore);
    uint64_t bigCoreNum = totalSrcTail - srcTailSmallCore * usedCoreNum;

    if (tailLen <= TAIL_LEN_THRESHOLD) {
        tilingKey = TILINR_KEY_WITH_SMALL_TAIL;
    } else {
        tilingKey = TILINR_KEY_WITH_LARGE_TAIL;
    }

    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(tilingKey);
    tilingData.set_srcTailBigCore(srcTailBigCore);
    tilingData.set_srcTailSmallCore(srcTailSmallCore);
    tilingData.set_bigCoreNum(bigCoreNum);
    tilingData.set_tailLenThreshold(TAIL_LEN_THRESHOLD);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTilingFuncV3(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ScatterAddTilingV3 tilingData;
    return tilingData.SetKernelTilingData(context);
}
}


namespace ge {
static ge::graphStatus ScatterAddInferShapeV3(gert::InferShapeContext* context)
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

static ge::graphStatus ScatterAddInferDataTypeV3(gert::InferDataTypeContext *context)
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
class ScatterAddV3 : public OpDef {
public:
    explicit ScatterAddV3(const char* name) : OpDef(name)
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
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("dim").Int();

        this->SetInferShape(ge::ScatterAddInferShapeV3)
            .SetInferDataType(ge::ScatterAddInferDataTypeV3);

        this->AICore().SetTiling(optiling::ScatterAddTilingFuncV3);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(ScatterAddV3);
}
