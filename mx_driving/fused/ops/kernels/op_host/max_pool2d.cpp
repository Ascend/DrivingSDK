/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "max_pool2d.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace ge;
using namespace std;
using namespace AscendC;

namespace {
const uint32_t BATCH_DIM = 0;
const uint32_t CHANNEL_DIM = 3;
const uint32_t HEIGHT_DIM = 1;
const uint32_t WIDTH_DIM = 2;
} // namespace

namespace optiling {
static ge::graphStatus TilingFuncForMaxPool2d(gert::TilingContext *context)
{
    MaxPool2dTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto xTensorPtr = context->GetInputTensor(0);
    if (xTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto xShape = xTensorPtr->GetStorageShape();

    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);

    uint32_t coreNum = ascendplatformInfo.GetCoreNumAiv();
    context->SetBlockDim(coreNum);

    tiling.set_batchSize(xShape.GetDim(BATCH_DIM));
    tiling.set_channel(xShape.GetDim(CHANNEL_DIM));
    tiling.set_inHeight(xShape.GetDim(HEIGHT_DIM));
    tiling.set_inWidth(xShape.GetDim(WIDTH_DIM));
    tiling.set_outHeight((xShape.GetDim(HEIGHT_DIM) + 1)/2);
    tiling.set_outWidth((xShape.GetDim(WIDTH_DIM) + 1)/2);

    tiling.set_coreNum(coreNum);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForMaxPool2d(gert::InferShapeContext *context)
{
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    if (x_shape == nullptr || y_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto batch = x_shape->GetDim(0);
    auto height = x_shape->GetDim(1);
    auto width = x_shape->GetDim(2);
    auto channel = x_shape->GetDim(3);

    y_shape->SetDimNum(0);
    y_shape->AppendDim(batch);
    y_shape->AppendDim((height + 1)/2);
    y_shape->AppendDim((width + 1)/2);
    y_shape->AppendDim(channel);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMaxPool2d(gert::InferDataTypeContext *context)
{
    const ge::DataType value_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, value_dtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class MaxPool2d : public OpDef {
public:
    explicit MaxPool2d(const char *name) : OpDef(name)
    {
        this->Input("x_trans")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y_trans")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShapeForMaxPool2d)
            .SetInferDataType(ge::InferDataTypeForMaxPool2d);
        this->AICore().SetTiling(optiling::TilingFuncForMaxPool2d);

        OpAICoreConfig aiConfig;
        aiConfig.ExtendCfgInfo("enableVectorCore.flag", "false");
        aiConfig.DynamicCompileStaticFlag(true);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend310p", aiConfig);
    }
};

OP_ADD(MaxPool2d);
} // namespace ops
