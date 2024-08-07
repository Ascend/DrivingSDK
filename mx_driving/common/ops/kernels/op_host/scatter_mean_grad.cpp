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
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t MASK_BYTES = 256;
constexpr uint64_t RESERVE_SAPCE = 2 * 1024;
constexpr uint32_t FLOAT_DTYPE_BYTES = 4;

class ScatterMeanGradTiling {
public:
    explicit ScatterMeanGradTiling(gert::TilingContext* context) : context(context){};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint();
private:
    void SetTilingKeyMode(int32_t axis, int32_t gradDims, int32_t indexDims, ge::DataType gradDtype, uint32_t coreNum);
    ScatterMeanGradTilingData TilingData;
    gert::TilingContext* context = nullptr;
    uint32_t paramsPre = 1;
    uint32_t dimRange = 1;
    uint32_t dimRangeOut = 1;
    uint32_t paramsPro = 1;
    int32_t dim = 0;
    
    uint32_t taskPerCore = 1;
    uint32_t coreUsed = 1;
    uint32_t taskTailCore = 1;

    uint64_t ubSize = 192 * 1024;
    uint32_t gradInUbSize = 1;
    uint32_t indexUbSize = 1;
    uint32_t gradOutUbSize = 1;
    uint32_t indexSumUbSize = 1;
    uint32_t gradInNum = 1;
    uint32_t indexNum = 1;
    uint32_t gradOutNum = 1;

    uint32_t gradDsize;
    uint32_t paramsNumPerMask = 1;
    uint32_t paramsNumPerBlock = 1;
    uint32_t indexNumPerBlock = 8;
    uint32_t indexNumPerMask = 64;
};

uint32_t CeilValue(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return 0;
    }
    return ((a - 1) / b + 1) * b;
}

void ScatterMeanGradTiling::SetTilingKeyMode(int32_t axis, int32_t gradDims, int32_t indexDims,
                                             ge::DataType gradDtype, uint32_t coreNum)
{
    uint32_t allDataSize = (dimRange + dimRangeOut) * paramsPro * sizeof(int32_t) + (2 * dimRangeOut + dimRange) * paramsPro * gradDsize;
    uint32_t lineDataSize = (dimRange + dimRangeOut) * sizeof(int32_t) + (3 * dimRangeOut + dimRange) * gradDsize;
    coreNum = coreNum == 0 ? 1 : coreNum;
    uint32_t availableSize = ubSize - RESERVE_SAPCE;
    if (gradDims == indexDims && allDataSize <= availableSize) {
        context->SetTilingKey(DATA_SMALL_MODE);
        gradInUbSize = CeilValue(dimRange * paramsPro, paramsNumPerBlock);
        indexUbSize = CeilValue(dimRange * paramsPro, indexNumPerBlock);
        uint32_t outputNum = CeilValue(dimRangeOut * paramsPro, paramsNumPerMask);
        gradOutUbSize = outputNum;
        indexSumUbSize = outputNum;
        taskPerCore = paramsPre / coreNum;
        coreUsed = paramsPre >= coreNum ? coreNum : paramsPre;
        taskTailCore = paramsPre % coreNum;
    } else if ((gradDims > indexDims) && (axis == indexDims - 1) && (lineDataSize <= availableSize)) {
        context->SetTilingKey(NOT_BROAD_LINE_MODE);
        gradInUbSize = gradOutUbSize = CeilValue(paramsPro, paramsNumPerBlock);
        indexUbSize = CeilValue(dimRange, indexNumPerBlock);
        taskPerCore = dimRangeOut / coreNum;
        coreUsed = dimRangeOut >= coreNum ? coreNum : dimRangeOut;
        taskTailCore = dimRangeOut % coreNum;
        indexSumUbSize = CeilValue(dimRangeOut, indexNumPerMask);
    }
}

ge::graphStatus ScatterMeanGradTiling::Init()
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();

    uint64_t totalUbSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, totalUbSize);
    ubSize = totalUbSize - RESERVE_SAPCE;

    auto gradOutShape = context->GetInputShape(0)->GetStorageShape();
    auto indexShape = context->GetInputShape(1)->GetStorageShape();
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
    for (int32_t i = axis + 1; i < gradDims; i++) {
        paramsPro *= gradInShape.GetDim(i);
    }
    gradInNum = paramsPre * dimRange * paramsPro;
    for (int32_t i = 0; i < indexDims; i++) {
        indexNum *= indexShape.GetDim(i);
    }
    gradOutNum = paramsPre * dimRangeOut * paramsPro;

    auto gradDtype = context->GetInputDesc(0)->GetDataType();
    gradDsize = sizeof(gradDtype);
    paramsNumPerMask = MASK_BYTES / sizeof(gradDtype);
    paramsNumPerBlock = BLOCK_BYTES / gradDsize;
    SetTilingKeyMode(axis, gradDims, indexDims, gradDtype, coreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterMeanGradTiling::RunKernelTiling()
{
    context->SetBlockDim(coreUsed);
    TilingData.set_paramsPre(paramsPre);
    TilingData.set_dimRange(dimRange);
    TilingData.set_dimRangeOut(dimRangeOut);
    TilingData.set_paramsPro(paramsPro);
    TilingData.set_dim(dim);
    
    TilingData.set_taskPerCore(taskPerCore);
    TilingData.set_taskTailCore(taskTailCore);

    TilingData.set_ubSize(ubSize);
    TilingData.set_gradInUbSize(gradInUbSize);
    TilingData.set_indexUbSize(indexUbSize);
    TilingData.set_gradOutUbSize(gradOutUbSize);
    TilingData.set_indexSumUbSize(indexSumUbSize);
    
    TilingData.set_gradInNum(gradInNum);
    TilingData.set_indexNum(indexNum);
    TilingData.set_gradOutNum(gradOutNum);

    size_t sysWorkspaceSize = WORKSPACE_16MBYTE_SIZE;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
    TilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc4ScatterMeanGrad(gert::TilingContext* context)
{
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
        this->AICore().AddConfig("ascend910c");
    }
};

OP_ADD(ScatterMeanGrad);
}