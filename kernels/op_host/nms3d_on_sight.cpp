/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "nms3d_on_sight_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <log/log.h>
#include "csrc/utils.h"
using namespace std;

namespace {
constexpr uint32_t BLOCK = 256;
} // namespace

namespace optiling {

static ge::graphStatus Nms3dOnSightTilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    Nms3dOnSightTilingData tiling;
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();

    if (context->GetInputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputDesc(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto boxShape = context->GetInputShape(0)->GetStorageShape(); // [7, N]
    auto maskShape = context->GetOutputShape(0)->GetStorageShape(); // [N, N_aligned]
    auto dtype = context->GetInputDesc(0)->GetDataType();
    auto attrs = context->GetAttrs();

    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 预留fp16的接口支持，目前不支持
    if (ge::DT_FLOAT == dtype) {
        context->SetTilingKey(1);
    } else if (ge::DT_FLOAT16 == dtype) {
        context->SetTilingKey(2);
    } else {
        return ge::GRAPH_FAILED;
    }

    uint32_t boxNum = boxShape.GetDim(1);
    uint32_t alignedN = maskShape.GetDim(1);
    uint32_t assignBox = AlignUp(alignedN, BLOCK / sizeof(dtype));

    // 按照boxNum进行分核，做拦截保护,需要注意两种OOM问题的拦截保护
    uint32_t usedCoreNum = std::min(boxNum, coreNum);
    uint32_t loopTime = 0;
    if (usedCoreNum != 0) {
        loopTime = (boxNum + usedCoreNum - 1) / usedCoreNum;
    } else {
        return ge::GRAPH_FAILED;
    }
    float threshold = *(attrs->GetAttrPointer<float>(0));

    context->SetBlockDim(usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_boxNum(boxNum);
    tiling.set_loopTime(loopTime);
    tiling.set_assignBox(assignBox);
    tiling.set_alignedN(alignedN);
    tiling.set_threshold(threshold);
    MX_DRIVING_LOGI("Nms3dOnSight tiling: usedCoreNum=%d, boxNum=%d, loopTime=%d, alignedN=%d, threshold=%f, assignBox=%d",
        usedCoreNum, boxNum, loopTime, alignedN, threshold, assignBox);

    // 待拆解功能
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus Nms3dOnSightInferShape(gert::InferShapeContext *context)
{
    return GRAPH_SUCCESS;
}
static ge::graphStatus Nms3dOnSightInferDataType(gert::InferDataTypeContext *context)
{
    context -> SetOutputDataType(0, ge::DT_INT16);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Nms3dOnSight : public OpDef {
public:
    explicit Nms3dOnSight(const char *name) : OpDef(name)
    {
        this->Input("boxes")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("mask")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT16})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("threshold").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::Nms3dOnSightInferShape)
            .SetInferDataType(ge::Nms3dOnSightInferDataType);

        this->AICore().SetTiling(optiling::Nms3dOnSightTilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(Nms3dOnSight);
} // namespace ops
