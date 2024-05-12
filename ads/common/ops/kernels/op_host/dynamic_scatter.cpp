/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#include "dynamic_scatter_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace ge;
using namespace std;
using namespace AscendC;

namespace optiling {
constexpr uint32_t DIM_INDEX0 = 0;
constexpr uint32_t DIM_INDEX1 = 1;

static std::map<std::string, uint32_t> REDUCE_TYPE_MAP = {{"max", 0}, {"sum", 1}};
class DynamicScatterTiling {
public:
    explicit DynamicScatterTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();

private:
    uint32_t GetNeedCoreNum(const uint32_t coreNumPlatform) const;

private:
    DynamicScatterTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    uint32_t pointNum;
    uint32_t featsNum;
    uint32_t coreNum;
    uint32_t outNum;
    uint32_t reduceMode;
};

uint32_t DynamicScatterTiling::GetNeedCoreNum(const uint32_t coreNumPlatform) const
{
    uint32_t tempCoreNum = pointNum;
    if (tempCoreNum == 0) {
        tempCoreNum = 1;
    }
    if (tempCoreNum < coreNumPlatform) {
        return tempCoreNum;
    } else {
        return coreNumPlatform;
    }
}

ge::graphStatus DynamicScatterTiling::Init()
{
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* currentWorkSpace = tilingContext->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;
    auto featsShape = tilingContext->GetInputShape(0)->GetStorageShape();
    pointNum = featsShape.GetDim(DIM_INDEX0);
    featsNum = featsShape.GetDim(DIM_INDEX1);
    auto reducedFeatsShape = tilingContext->GetOutputShape(0)->GetStorageShape();
    outNum = reducedFeatsShape.GetDim(DIM_INDEX0);

    auto platformInfo = tilingContext->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    coreNum = GetNeedCoreNum(coreNum);

    auto attrs = tilingContext->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const char* reduceTypePtr = attrs->GetAttrPointer<char>(DIM_INDEX0);
    std::string reduceType(reduceTypePtr);
    if (reduceType != "max" && reduceType != "sum") {
        return ge::GRAPH_PARAM_INVALID;
    }

    reduceMode = REDUCE_TYPE_MAP[reduceType];
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicScatterTiling::RunKernelTiling()
{
    tilingContext->SetBlockDim(coreNum);
    tilingData.set_featsNum(featsNum);
    tilingData.set_pointNum(pointNum);
    tilingData.set_coreNum(coreNum);
    tilingData.set_outNum(outNum);
    tilingData.set_reduceMode(reduceMode);

    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingDynamicScatter(gert::TilingContext* context)
{
    DynamicScatterTiling tilingObject(context);
    tilingObject.Init();
    return tilingObject.RunKernelTiling();
}
} // namespace optiling


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* featShape = context->GetInputShape(0);
    if (featShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outShape->SetDim(0, -1);
    outShape->SetDim(1, featShape->GetDim(1));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class DynamicScatter : public OpDef {
public:
    explicit DynamicScatter(const char* name) : OpDef(name)
    {
        this->Input("feats")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("coors_map")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("reduced_feats")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("reduce_type").AttrType(REQUIRED).String("max");
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingDynamicScatter);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(DynamicScatter);
} // namespace ops
