/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#include "scatter_max_v3.h"
#include "common.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t UB_PRESERVED = 1024;
constexpr uint64_t UB_PRESERVED_ARGMAX = 10 * 1024;
constexpr uint64_t MAX_BATCH_NUM = 4095;
constexpr uint64_t ARGAMX_PARALLEL_DEGREE = 3;
constexpr uint64_t TILING_KEY_SMALL_TAIL = 0;
constexpr uint64_t TILING_KEY_LARGE_TAIL = 1;

namespace optiling {

class ScatterMaxV3Tiling {
public:
    ScatterMaxV3Tiling() {}
    ScatterMaxTilingDataV3* GetTilingData(gert::TilingContext*);
    ScatterMaxTilingDataV3* GetArgmaxTilingData(gert::TilingContext*);

private:
    ge::graphStatus init(gert::TilingContext*);
    ge::graphStatus setTilingData();

private:
    ScatterMaxTilingDataV3 tiling;

    uint64_t ubSize;
    uint64_t coreNum;
    uint64_t srcElemNum, idxElemNum, resElemNum, tailElemNum;
    uint64_t srcDSize, idxDSize, tailSize, tailSizeAlign;
    uint64_t elemNumPerBlock;

    uint64_t idxNumPerCore, idxBatchNum;
    // small tail
    uint64_t tailBatchNum;
    // large tail 
    uint64_t srcBatchNum;
    // left
    uint64_t coreNumPerTail, leftSrcBatchNum;
    uint64_t leftSrcNumBigCore, leftSrcBigCoreNum;
};

ge::graphStatus ScatterMaxV3Tiling::init(gert::TilingContext* ctx)
{
    if (ctx == nullptr || ctx->GetInputDesc(0) == nullptr || ctx->GetInputDesc(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto srcDtype = ctx->GetInputDesc(0)->GetDataType();
    auto idxDtype = ctx->GetInputDesc(1)->GetDataType();
    srcDSize = kDataSizeMap[srcDtype]; // now only support float32
    idxDSize = kDataSizeMap[idxDtype]; // now only support int32

    if (srcDSize == 0 || idxDSize == 0) {
        return ge::GRAPH_FAILED;
    }
    elemNumPerBlock = BLOCK_SIZE / srcDSize;

    if (ctx->GetInputShape(0) == nullptr || ctx->GetInputShape(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto srcShape = ctx->GetInputShape(0)->GetStorageShape();
    auto idxShape = ctx->GetInputShape(1)->GetStorageShape();
    auto resShape = ctx->GetOutputShape(0)->GetStorageShape();

    srcElemNum = srcShape.GetShapeSize();
    idxElemNum = idxShape.GetShapeSize();
    resElemNum = resShape.GetShapeSize();
    tailElemNum = srcElemNum / srcShape.GetDim(0);

    tailSize = tailElemNum * srcDSize;
    tailSizeAlign = CeilAlign(tailElemNum, elemNumPerBlock) * srcDSize;

    auto platformInfo = ctx->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatfrom = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = ascendcPlatfrom.GetCoreNumAiv();

    ascendcPlatfrom.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    return ge::GRAPH_SUCCESS;
}

 ge::graphStatus ScatterMaxV3Tiling::setTilingData()
 {
    tiling.set_srcElemNum(srcElemNum);
    tiling.set_idxElemNum(idxElemNum);
    tiling.set_resElemNum(resElemNum);
    tiling.set_tailElemNum(tailElemNum);
    tiling.set_tailSize(tailSize);
    tiling.set_elemNumPerBlock(elemNumPerBlock);
    tiling.set_idxNumPerCore(idxNumPerCore);
    tiling.set_idxBatchNum(idxBatchNum);
    tiling.set_tailBatchNum(tailBatchNum);
    tiling.set_srcBatchNum(srcBatchNum);
    tiling.set_coreNumPerTail(coreNumPerTail);
    tiling.set_leftSrcNumBigCore(leftSrcNumBigCore);
    tiling.set_leftSrcBigCoreNum(leftSrcBigCoreNum);
    tiling.set_leftSrcBatchNum(leftSrcBatchNum);

    return ge::GRAPH_SUCCESS;
}

ScatterMaxTilingDataV3* ScatterMaxV3Tiling::GetTilingData(gert::TilingContext* ctx)
{
    if (init(ctx) == ge::GRAPH_FAILED) {
        return nullptr;
    }

    ubSize -= UB_PRESERVED;

    idxNumPerCore = idxElemNum / coreNum;
    idxBatchNum = std::min(idxNumPerCore, MAX_BATCH_NUM);

    uint64_t remainUbSize = ubSize - CeilAlign(idxBatchNum, elemNumPerBlock) * idxDSize;
    tailBatchNum = remainUbSize / tailSizeAlign;
    srcBatchNum = remainUbSize / BLOCK_SIZE * elemNumPerBlock;

    uint64_t leftIdxNum = idxElemNum % coreNum;
    if (leftIdxNum == 0) {
        coreNumPerTail = 0;
        leftSrcNumBigCore = 0;
        leftSrcBigCoreNum = 0;
        leftSrcBatchNum = 0;
    } else {
        coreNumPerTail = std::min(coreNum / leftIdxNum, tailElemNum);
        leftSrcNumBigCore = DivCeil(tailElemNum, coreNumPerTail);
        leftSrcBigCoreNum = tailElemNum - coreNumPerTail * (leftSrcNumBigCore - 1);
        leftSrcBatchNum = ubSize / BLOCK_SIZE * elemNumPerBlock;
    }

    ctx->SetBlockDim(coreNum);
    if (idxNumPerCore != 0 && tailBatchNum == 0) {
        ctx->SetTilingKey(TILING_KEY_LARGE_TAIL);
    } else {
        ctx->SetTilingKey(TILING_KEY_SMALL_TAIL);
    }

    setTilingData();

    return &tiling;
}

ScatterMaxTilingDataV3* ScatterMaxV3Tiling::GetArgmaxTilingData(gert::TilingContext* ctx)
{
    if (init(ctx) == ge::GRAPH_FAILED) {
        return nullptr;
    }

    ubSize -= UB_PRESERVED_ARGMAX;

    idxNumPerCore = idxElemNum / coreNum;
    idxBatchNum = std::min(idxNumPerCore, MAX_BATCH_NUM);

    uint64_t remainUbSize = ubSize - CeilAlign(idxBatchNum, elemNumPerBlock) * idxDSize;
    tailBatchNum = remainUbSize / tailSizeAlign;
    srcBatchNum = remainUbSize / BLOCK_SIZE / ARGAMX_PARALLEL_DEGREE * elemNumPerBlock;

    uint64_t leftIdxNum = idxElemNum % coreNum;
    if (leftIdxNum == 0) {
        coreNumPerTail = 0;
        leftSrcNumBigCore = 0;
        leftSrcBigCoreNum = 0;
        leftSrcBatchNum = 0;
    } else {
        coreNumPerTail = std::min(coreNum / leftIdxNum, tailElemNum);
        leftSrcNumBigCore = DivCeil(tailElemNum, coreNumPerTail);
        leftSrcBigCoreNum = tailElemNum - coreNumPerTail * (leftSrcNumBigCore - 1);
        leftSrcBatchNum = ubSize / BLOCK_SIZE / ARGAMX_PARALLEL_DEGREE * elemNumPerBlock;
    }

    ctx->SetBlockDim(coreNum);
    if (idxNumPerCore != 0 && tailBatchNum < ARGAMX_PARALLEL_DEGREE) {
        ctx->SetTilingKey(TILING_KEY_LARGE_TAIL);
    } else {
        tailBatchNum -= (ARGAMX_PARALLEL_DEGREE - 1);
        ctx->SetTilingKey(TILING_KEY_SMALL_TAIL);
    }

    setTilingData();

    return &tiling;
}
}

namespace optiling {
template<bool argmax>
static ge::graphStatus ScatterMaxV3TilingFunc(gert::TilingContext* ctx)
{
    ScatterMaxV3Tiling tiling;
    ScatterMaxTilingDataV3* tilingData;

    if constexpr (argmax) {
        tilingData = tiling.GetArgmaxTilingData(ctx);
    } else {
        tilingData = tiling.GetTilingData(ctx);
    }

    if (tilingData == nullptr) {
        return ge::GRAPH_FAILED;
    }

    if (ctx->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tilingData->SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
    ctx->GetRawTilingData()->SetDataSize(tilingData->GetDataSize());

    size_t *currentWorkspace = ctx->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus ScatterMaxV3InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    gert::Shape* argmax_shape = context->GetOutputShape(1);
    if (x1_shape == nullptr || y_shape == nullptr || argmax_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *y_shape = *x1_shape;
    *argmax_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus ScatterMaxV3InferDtype(gert::InferDataTypeContext *context)
{
    const ge::DataType var_dtype = context->GetInputDataType(0);
    const ge::DataType indices_dtype = context->GetInputDataType(1);
    context->SetOutputDataType(0, var_dtype);
    context->SetOutputDataType(1, indices_dtype);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class ScatterMaxV3 : public OpDef {
public:
    explicit ScatterMaxV3(const char* name) : OpDef(name)
    {
        this->Input("src")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("res")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("argmax")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::ScatterMaxV3InferShape)
             .SetInferDataType(ge::ScatterMaxV3InferDtype);

        this->AICore()
            .SetTiling(optiling::ScatterMaxV3TilingFunc<false>);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(ScatterMaxV3);
}


namespace ops {
class ScatterMaxArgmaxV3 : public OpDef {
public:
    explicit ScatterMaxArgmaxV3(const char* name) : OpDef(name)
    {
        this->Input("src")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("res")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("argmax")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::ScatterMaxV3InferShape)
             .SetInferDataType(ge::ScatterMaxV3InferDtype);

        this->AICore()
            .SetTiling(optiling::ScatterMaxV3TilingFunc<true>);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(ScatterMaxArgmaxV3);
}