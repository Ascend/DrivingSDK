/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
 */
#include "boxes_overlap_bev_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "ge/utils.h"

using namespace ge;
using namespace std;

namespace {
const uint32_t POS_INPUT_BOXES_A = 0;
const uint32_t POS_INPUT_BOXES_B = 1;

const uint32_t POS_OUTPUT_RES = 0;

const uint32_t POS_ATTR_FORMAT_FLAG = 0;
const uint32_t POS_ATTR_CLOCKWISE = 1;
const uint32_t POS_ATTR_MODE_FLAG = 2;
const uint32_t POS_ATTR_ALIGNED = 3;
const uint32_t POS_ATTR_MARGIN = 4;

const uint32_t BOXES_NUM_DIM = 0;
const uint32_t BOXES_FORMAT_SIZE_DIM = 1;

const uint32_t TILING_KEY_TT = 0;
const uint32_t TILING_KEY_TF = 1;
const uint32_t TILING_KEY_FT = 2;
const uint32_t TILING_KEY_FF = 3;
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc4BoxesOverlapBev(gert::TilingContext *context)
{
    BoxesOverlapBevTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(coreNum);

    auto boxesATensorPtr = context->GetInputTensor(POS_INPUT_BOXES_A);
    auto boxesBTensorPtr = context->GetInputTensor(POS_INPUT_BOXES_B);
    if (boxesATensorPtr == nullptr || boxesBTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto formatFlagPtr = attrs->GetAttrPointer<int>(POS_ATTR_FORMAT_FLAG);
    auto clockwisePtr = attrs->GetAttrPointer<bool>(POS_ATTR_CLOCKWISE);
    auto modeFlagPtr = attrs->GetAttrPointer<int>(POS_ATTR_MODE_FLAG);
    auto alignedPtr = attrs->GetAttrPointer<bool>(POS_ATTR_ALIGNED);
    auto marginPtr = attrs->GetAttrPointer<float>(POS_ATTR_MARGIN);
    if (formatFlagPtr == nullptr || clockwisePtr == nullptr || modeFlagPtr == nullptr ||
        alignedPtr == nullptr || marginPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto formatFlag = *formatFlagPtr;
    auto clockwise = *clockwisePtr;
    auto modeFlag = *modeFlagPtr;
    auto aligned = *alignedPtr;
    auto margin = *marginPtr;

    auto boxesAShape = boxesATensorPtr->GetStorageShape();
    auto boxesBShape = boxesBTensorPtr->GetStorageShape();

    uint32_t boxesANum = boxesAShape.GetDim(BOXES_NUM_DIM);
    uint32_t boxesBNum = boxesBShape.GetDim(BOXES_NUM_DIM);
    uint32_t boxesFormatSize = boxesAShape.GetDim(BOXES_FORMAT_SIZE_DIM);

    uint32_t boxesMinNum = boxesANum < boxesBNum ? boxesANum : boxesBNum;
    uint64_t taskNum = aligned ? boxesMinNum : static_cast<uint64_t>(boxesANum) * boxesBNum;

    uint32_t numLargeCores = static_cast<uint32_t>(taskNum % coreNum);
    if (numLargeCores == 0) {
        numLargeCores = coreNum;
    }
    uint64_t numTasksPerLargeCore = Ceil(taskNum, coreNum);

    tiling.set_boxesANum(boxesANum);
    tiling.set_boxesBNum(boxesBNum);
    tiling.set_boxesFormatSize(boxesFormatSize);
    tiling.set_numLargeCores(numLargeCores);
    tiling.set_numTasksPerLargeCore(numTasksPerLargeCore);
    tiling.set_formatFlag(formatFlag);
    tiling.set_modeFlag(modeFlag);
    tiling.set_margin(margin);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    if (clockwise && aligned) {
        context->SetTilingKey(TILING_KEY_TT);
    } else if (clockwise && !aligned) {
        context->SetTilingKey(TILING_KEY_TF);
    } else if (!clockwise && aligned) {
        context->SetTilingKey(TILING_KEY_FT);
    } else if (!clockwise && !aligned) {
        context->SetTilingKey(TILING_KEY_FF);
    }

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus Infershape4BoxesOverlapBev(gert::InferShapeContext *context)
{
    auto boxesAShape = context->GetInputShape(POS_INPUT_BOXES_A);
    auto boxesBShape = context->GetInputShape(POS_INPUT_BOXES_B);
    auto areaOverlapShape = context->GetOutputShape(POS_OUTPUT_RES);
    if (boxesAShape == nullptr || boxesBShape == nullptr || areaOverlapShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto boxesANum = boxesAShape->GetDim(BOXES_NUM_DIM);
    auto boxesBNum = boxesBShape->GetDim(BOXES_NUM_DIM);

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto alignedPtr = attrs->GetAttrPointer<bool>(POS_ATTR_ALIGNED);
    if (alignedPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto aligned = *alignedPtr;

    if (aligned) {
        auto boxesMinNum = boxesANum < boxesBNum ? boxesANum : boxesBNum;
        areaOverlapShape->SetDimNum(0);
        areaOverlapShape->AppendDim(boxesMinNum);
    } else {
        areaOverlapShape->SetDimNum(0);
        areaOverlapShape->AppendDim(boxesANum);
        areaOverlapShape->AppendDim(boxesBNum);
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4BoxesOverlapBev(gert::InferDataTypeContext *context)
{
    const ge::DataType box_dtype = context->GetInputDataType(POS_INPUT_BOXES_A);
    context->SetOutputDataType(POS_OUTPUT_RES, box_dtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class BoxesOverlapBev : public OpDef {
public:
    explicit BoxesOverlapBev(const char *name) : OpDef(name)
    {
        this->Input("boxes_a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("boxes_b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("res")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("format_flag").AttrType(OPTIONAL).Int(1);
        this->Attr("clockwise").AttrType(OPTIONAL).Bool(true);
        this->Attr("mode_flag").AttrType(OPTIONAL).Int(0);
        this->Attr("aligned").AttrType(OPTIONAL).Bool(false);
        this->Attr("margin").AttrType(OPTIONAL).Float(1e-5);

        this->SetInferShape(ge::Infershape4BoxesOverlapBev)
            .SetInferDataType(ge::InferDataType4BoxesOverlapBev);

        this->AICore().SetTiling(optiling::TilingFunc4BoxesOverlapBev);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(BoxesOverlapBev);
} // namespace ops
