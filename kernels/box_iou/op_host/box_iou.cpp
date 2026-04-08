/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
 */
#include "box_iou_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace ge;
using namespace std;

namespace {
const uint32_t POS_INPUT_BOXES_A = 0;
const uint32_t POS_INPUT_BOXES_B = 1;
const uint32_t POS_OUTPUT_IOUS = 0;
const uint32_t POS_ATTR_MODE_FLAG = 0;
const uint32_t POS_ATTR_ALIGNED = 1;
const uint32_t BOXES_NUM_DIM = 0;
const uint32_t BOXES_DESC_DIM = 1;
const uint32_t TILING_KEY_ROTATED_ALIGNED_IOU = 0; // box_iou_rotated, aligned=true, modeFlag=0
const uint32_t TILING_KEY_ROTATED_ALIGNED_IOF = 1; // box_iou_rotated, aligned=true, modeFlag=1
const uint32_t TILING_KEY_ROTATED_UNALIGNED_IOU = 2; // box_iou_rotated, aligned=false, modeFlag=0
const uint32_t TILING_KEY_ROTATED_UNALIGNED_IOF = 3; // box_iou_rotated, aligned=false, modeFlag=1
const uint32_t TILING_KEY_QUADRI_ALIGNED_IOU = 4; // box_iou_quadri, aligned=true, modeFlag=0
const uint32_t TILING_KEY_QUADRI_ALIGNED_IOF = 5; // box_iou_quadri, aligned=true, modeFlag=1
const uint32_t TILING_KEY_QUADRI_UNALIGNED_IOU = 6; // box_iou_quadri, aligned=false, modeFlag=0
const uint32_t TILING_KEY_QUADRI_UNALIGNED_IOF = 7; // box_iou_quadri, aligned=false, modeFlag=1

uint32_t DivCeil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc4BoxIou(gert::TilingContext *context)
{
    BoxIouTilingData tiling;
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
    auto modeFlagPtr = attrs->GetAttrPointer<int>(POS_ATTR_MODE_FLAG);
    auto alignedPtr = attrs->GetAttrPointer<bool>(POS_ATTR_ALIGNED);
    if (modeFlagPtr == nullptr || alignedPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto modeFlag = *modeFlagPtr;
    auto aligned = *alignedPtr;

    auto boxesAShape = boxesATensorPtr->GetStorageShape();
    auto boxesBShape = boxesBTensorPtr->GetStorageShape();

    auto boxesANum = boxesAShape.GetDim(BOXES_NUM_DIM);
    auto boxesBNum = boxesBShape.GetDim(BOXES_NUM_DIM);
    auto boxesDescDimNum = boxesAShape.GetDim(BOXES_DESC_DIM);

    auto boxesMinNum = boxesANum < boxesBNum ? boxesANum : boxesBNum;
    auto boxesMaxNum = boxesANum > boxesBNum ? boxesANum : boxesBNum;

    auto taskNum = aligned ? boxesMinNum : boxesMaxNum;
    auto taskNumPerCore = DivCeil(taskNum, coreNum);
    auto outerLoopCnt = taskNum;
    auto innerLoopCnt = boxesMinNum;

    tiling.set_boxesANum(boxesANum);
    tiling.set_boxesBNum(boxesBNum);
    tiling.set_taskNum(taskNum);
    tiling.set_taskNumPerCore(taskNumPerCore);
    tiling.set_outerLoopCnt(outerLoopCnt);
    tiling.set_innerLoopCnt(innerLoopCnt);
    tiling.set_boxesDescDimNum(boxesDescDimNum);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    if (boxesDescDimNum == 5) {
        if (aligned && modeFlag == 0) {
            context->SetTilingKey(TILING_KEY_ROTATED_ALIGNED_IOU);
        } else if (aligned && modeFlag == 1) {
            context->SetTilingKey(TILING_KEY_ROTATED_ALIGNED_IOF);
        } else if (!aligned && modeFlag == 0) {
            context->SetTilingKey(TILING_KEY_ROTATED_UNALIGNED_IOU);
        } else if (!aligned && modeFlag == 1) {
            context->SetTilingKey(TILING_KEY_ROTATED_UNALIGNED_IOF);
        }
    } else if (boxesDescDimNum == 8) {
        if (aligned && modeFlag == 0) {
            context->SetTilingKey(TILING_KEY_QUADRI_ALIGNED_IOU);
        } else if (aligned && modeFlag == 1) {
            context->SetTilingKey(TILING_KEY_QUADRI_ALIGNED_IOF);
        } else if (!aligned && modeFlag == 0) {
            context->SetTilingKey(TILING_KEY_QUADRI_UNALIGNED_IOU);
        } else if (!aligned && modeFlag == 1) {
            context->SetTilingKey(TILING_KEY_QUADRI_UNALIGNED_IOF);
        }
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
static ge::graphStatus Infershape4BoxIou(gert::InferShapeContext *context)
{
    auto boxesAShape = context->GetInputShape(POS_INPUT_BOXES_A);
    auto boxesBShape = context->GetInputShape(POS_INPUT_BOXES_B);
    auto iousShape = context->GetOutputShape(POS_OUTPUT_IOUS);
    if (boxesAShape == nullptr || boxesBShape == nullptr || iousShape == nullptr) {
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
        iousShape->SetDimNum(0);
        iousShape->AppendDim(boxesMinNum);
    } else {
        iousShape->SetDimNum(0);
        iousShape->AppendDim(boxesANum);
        iousShape->AppendDim(boxesBNum);
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4BoxIou(gert::InferDataTypeContext *context)
{
    const ge::DataType box_dtype = context->GetInputDataType(POS_INPUT_BOXES_A);
    context->SetOutputDataType(POS_OUTPUT_IOUS, box_dtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class BoxIou : public OpDef {
public:
    explicit BoxIou(const char *name) : OpDef(name)
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
        this->Output("ious")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("mode_flag").AttrType(OPTIONAL).Int(0);
        this->Attr("aligned").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::Infershape4BoxIou)
            .SetInferDataType(ge::InferDataType4BoxIou);

        this->AICore().SetTiling(optiling::TilingFunc4BoxIou);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(BoxIou);
} // namespace ops
