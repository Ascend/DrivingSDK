/*
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#include "min_area_polygons_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
namespace optiling {
static uint32_t AlignUp(uint32_t x, uint32_t y)
{
    if (y == 0) {
        return x;
    }
    return (x -1 + y) / y;
}

static ge::graphStatus MinAreaPolygonsTilingFunc(gert::TilingContext* context)
{
    MinAreaPolygonsTilingData tiling;
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto pointsetsShape = context->GetInputShape(0)->GetStorageShape();
    if (context->GetInputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto dtype = context->GetInputDesc(0)->GetDataType();
    if (context->GetInputDesc(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t pointsetNum = pointsetsShape.GetDim(0);
    uint32_t coreTask = AlignUp(pointsetNum, coreNum);
    uint32_t usedCoreNum = AlignUp(pointsetNum, coreTask);
    uint32_t lastCoreTask = 0;
    if (coreTask != 0) {
        lastCoreTask = pointsetNum % coreTask;
    }
    if (lastCoreTask == 0) {
        lastCoreTask = coreTask;
    }
    if (ge::DT_FLOAT == dtype) {
        context->SetTilingKey(1);
    } else {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(usedCoreNum);
    tiling.set_pointsetNum(pointsetNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_coreTask(coreTask);
    tiling.set_lastCoreTask(lastCoreTask);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus MinAreaPolygonsInferShape(gert::InferShapeContext *context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMinAreaPolygons(gert::InferDataTypeContext *context)
{
    const ge::DataType valueDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, valueDtype);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class MinAreaPolygons : public OpDef {
public:
    explicit MinAreaPolygons(const char *name) : OpDef(name)
    {
        this->Input("pointsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("polygons")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::MinAreaPolygonsInferShape)
            .SetInferDataType(ge::InferDataTypeForMinAreaPolygons);

        this->AICore().SetTiling(optiling::MinAreaPolygonsTilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(MinAreaPolygons);
} // namespace ops