/*
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#include "draw_gaussian_to_heatmap_tiling.h"
#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
constexpr uint32_t RESERVED_UB_SIZE = 8 * 1024;

static ge::graphStatus TilingFuncForDrawGaussianToHeatmap(gert::TilingContext* context)
{
    CHECK_NULLPTR(context);
    auto platform = context->GetPlatformInfo();
    CHECK_NULLPTR(platform);
    auto platformInfo = platform_ascendc::PlatformAscendC(platform);
    uint32_t coreNum = platformInfo.GetCoreNumAiv();
    uint64_t ubSize;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    DrawGaussianToHeatmapTilingData tiling;
    auto attrsPtr = context->GetAttrs();
    CHECK_NULLPTR(attrsPtr);
    int64_t numClasses = *(attrsPtr->GetAttrPointer<int64_t>(0));
    int64_t featureMapSizeX = *(attrsPtr->GetAttrPointer<int64_t>(1));
    int64_t featureMapSizeY = *(attrsPtr->GetAttrPointer<int64_t>(2));
    auto centerIntPtr = context->GetInputTensor(2);
    CHECK_NULLPTR(centerIntPtr);
    auto centerIntShape = centerIntPtr->GetStorageShape();
    uint32_t taskObj = centerIntShape.GetDim(1);
    uint32_t coreTaskLen = Ceil(taskObj, coreNum);
    uint32_t usedCoreNum = Ceil(taskObj, coreTaskLen);
    uint32_t radiusLen = 1024 * 4 * sizeof(float);
    uint32_t singlePorcessCopyLen =  (ubSize - RESERVED_UB_SIZE - radiusLen) / 4 / 5;
    singlePorcessCopyLen = AlignUp(singlePorcessCopyLen, 32);
    uint32_t taskRepeatTimes = Ceil(coreTaskLen, singlePorcessCopyLen);
    context->SetBlockDim(usedCoreNum);
    tiling.set_coreTaskLen(coreTaskLen);
    tiling.set_numClasses(numClasses);
    tiling.set_taskObj(taskObj);
    tiling.set_taskRepeatTimes(taskRepeatTimes);
    tiling.set_singlePorcessCopyLen(singlePorcessCopyLen);
    tiling.set_featureMapSizeX(featureMapSizeX);
    tiling.set_featureMapSizeY(featureMapSizeY);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 16 * 1024 * 1024;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForDrawGaussianToHeatmap(gert::InferShapeContext* context)
{
    CHECK_NULLPTR(context);
    auto attrsPtr = context->GetAttrs();
    CHECK_NULLPTR(attrsPtr);
    gert::Shape* heatMapShape = context->GetOutputShape(0);
    CHECK_NULLPTR(heatMapShape);
    uint32_t numClasses = *(attrsPtr->GetAttrPointer<int32_t>(0));
    int64_t featureMapSizeX = *(attrsPtr->GetAttrPointer<int64_t>(1));
    int64_t featureMapSizeY = *(attrsPtr->GetAttrPointer<int64_t>(2));
    *heatMapShape = {numClasses, featureMapSizeY, featureMapSizeX};
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeForDrawGaussianToHeatmap(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class DrawGaussianToHeatmap : public OpDef {
public:
    explicit DrawGaussianToHeatmap(const char* name) : OpDef(name)
    {
        this->Input("mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cur_class_id")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("center_int")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("radius")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("heatmap")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("num_classes").AttrType(REQUIRED).Int();
        this->Attr("feature_map_size_x").AttrType(REQUIRED).Int();
        this->Attr("feature_map_size_y").AttrType(REQUIRED).Int();
        this->SetInferShape(ge::InferShapeForDrawGaussianToHeatmap).SetInferDataType(ge::InferDataTypeForDrawGaussianToHeatmap);

        this->AICore().SetTiling(optiling::TilingFuncForDrawGaussianToHeatmap);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(DrawGaussianToHeatmap);
} // namespace ops