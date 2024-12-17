/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "nms3d_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

using namespace std;

namespace optiling {
static ge::graphStatus Nms3dTilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    Nms3dTilingData tiling;
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();

    if (context->GetInputShape(0) == nullptr || context->GetOutputShape(0) == nullptr || context->GetInputDesc(0) == nullptr || context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto boxShape = context->GetInputShape(0)->GetStorageShape();
    auto maskShape = context->GetOutputShape(0)->GetStorageShape();
    auto dtype = context->GetInputDesc(0)->GetDataType();
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t boxNum = boxShape.GetDim(0);
    uint32_t maskNum = maskShape.GetDim(1);
    uint32_t dataAlign = 16;
    if (ge::DT_FLOAT == dtype) {
        context->SetTilingKey(1);
    } else if (ge::DT_FLOAT16 == dtype) {
        context->SetTilingKey(2);
    } else {
        return ge::GRAPH_FAILED;
    }

    uint32_t usedCoreNum = std::min((boxNum - 1) / dataAlign + 1, coreNum);
    uint32_t loopTime = (boxNum - 1) / (usedCoreNum * dataAlign) + 1;
    uint32_t tailSum = boxNum - usedCoreNum * (loopTime - 1) * dataAlign;
    uint32_t tailNum = (tailSum - 1) % dataAlign + 1;
    float nms_overlap_thresh = *(attrs->GetAttrPointer<float>(0));

    context->SetBlockDim(usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_boxNum(boxNum);
    tiling.set_loopTime(loopTime);
    tiling.set_eachSum(loopTime * dataAlign);
    tiling.set_tailSum(tailSum);
    tiling.set_tailNum(tailNum);
    tiling.set_maskNum(maskNum);
    tiling.set_overlapThresh(nms_overlap_thresh);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus Nms3dInferShape(gert::InferShapeContext *context)
{
    return GRAPH_SUCCESS;
}
static ge::graphStatus Nms3dInferDataType(gert::InferDataTypeContext *context)
{
    context -> SetOutputDataType(0, ge::DT_INT16);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Nms3d : public OpDef {
public:
    explicit Nms3d(const char *name) : OpDef(name)
    {
        this->Input("boxes")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("mask")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT16, ge::DT_INT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("threshold").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::Nms3dInferShape)
            .SetInferDataType(ge::Nms3dInferDataType);

        this->AICore().SetTiling(optiling::Nms3dTilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(Nms3d);
} // namespace ops
