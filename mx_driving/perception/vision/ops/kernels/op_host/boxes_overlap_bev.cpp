/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
 */
#include "boxes_overlap_bev_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace ge;
using namespace std;

namespace {
const uint32_t INPUT_BOXES_A = 0;
const uint32_t INPUT_BOXES_B = 1;
const uint32_t OUTPUT_AREA_OVERLAP = 0;
const uint32_t ATTR_TRANS = 0;
const uint32_t ATTR_IS_CLOCKWISE = 1;
const uint32_t ATTR_ALIGNED = 2;
const uint32_t ATTR_MODE_FLAG = 3;
const uint32_t BOXES_NUM_DIM = 0;
const uint32_t BOXES_DESC_DIM = 1;
const uint32_t TILING_KEY_ADS = 0; // ads boxes_overlap_bev
const uint32_t TILING_KEY_MMCV = 1; // mmcv boxes_overlap_bev
const uint32_t TILING_KEY_MMCV_BIR_ALIGNED_IOU = 2; // mmcv box_iou_rotated, aligned=true, modeFlag=0
const uint32_t TILING_KEY_MMCV_BIR_ALIGNED_IOF = 3; // mmcv box_iou_rotated, aligned=true, modeFlag=1
const uint32_t TILING_KEY_MMCV_BIR_UNALIGNED_IOU = 4; // mmcv box_iou_rotated, aligned=false, modeFlag=0
const uint32_t TILING_KEY_MMCV_BIR_UNALIGNED_IOF = 5; // mmcv box_iou_rotated, aligned=false, modeFlag=1


uint32_t DivCeil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
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

    auto boxesATensorPtr = context->GetInputTensor(INPUT_BOXES_A);
    auto boxesBTensorPtr = context->GetInputTensor(INPUT_BOXES_B);
    if (boxesATensorPtr == nullptr || boxesBTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto transPtr = attrs->GetAttrPointer<bool>(ATTR_TRANS);
    auto isClockwisePtr = attrs->GetAttrPointer<bool>(ATTR_IS_CLOCKWISE);
    auto alignedPtr = attrs->GetAttrPointer<bool>(ATTR_ALIGNED);
    auto modeFlagPtr = attrs->GetAttrPointer<int>(ATTR_MODE_FLAG);
    if (transPtr == nullptr || isClockwisePtr == nullptr || alignedPtr == nullptr || modeFlagPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto trans = *transPtr;
    auto isClockwise = *isClockwisePtr;
    auto aligned = *alignedPtr;
    auto modeFlag = *modeFlagPtr;

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
    tiling.set_trans(trans);
    tiling.set_isClockwise(isClockwise);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    if (!trans && boxesDescDimNum == 5) {
        if (aligned && modeFlag == 0) {
            context->SetTilingKey(TILING_KEY_MMCV_BIR_ALIGNED_IOU);
        } else if (aligned && modeFlag == 1) {
            context->SetTilingKey(TILING_KEY_MMCV_BIR_ALIGNED_IOF);
        } else if (!aligned && modeFlag == 0) {
            context->SetTilingKey(TILING_KEY_MMCV_BIR_UNALIGNED_IOU);
        } else if (!aligned && modeFlag == 1) {
            context->SetTilingKey(TILING_KEY_MMCV_BIR_UNALIGNED_IOF);
        }
    } else if (!trans && boxesDescDimNum == 7) {
        context->SetTilingKey(TILING_KEY_MMCV);
    } else {
        context->SetTilingKey(TILING_KEY_ADS);
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
    auto boxesAShape = context->GetInputShape(INPUT_BOXES_A);
    auto boxesBShape = context->GetInputShape(INPUT_BOXES_B);
    auto areaOverlapShape = context->GetOutputShape(OUTPUT_AREA_OVERLAP);
    if (boxesAShape == nullptr || boxesBShape == nullptr || areaOverlapShape) {
        return ge::GRAPH_FAILED;
    }
    auto boxesANum = boxesAShape->GetDim(BOXES_NUM_DIM);
    auto boxesBNum = boxesBShape->GetDim(BOXES_NUM_DIM);

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto alignedPtr = attrs->GetAttrPointer<bool>(ATTR_ALIGNED);
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
    const ge::DataType box_dtype = context->GetInputDataType(INPUT_BOXES_A);
    context->SetOutputDataType(OUTPUT_AREA_OVERLAP, box_dtype);
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
        this->Output("area_overlap")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("trans").AttrType(OPTIONAL).Bool(true);
        this->Attr("is_clockwise").AttrType(OPTIONAL).Bool(true);
        this->Attr("aligned").AttrType(OPTIONAL).Bool(false);
        this->Attr("mode_flag").AttrType(OPTIONAL).Int(2);

        this->SetInferShape(ge::Infershape4BoxesOverlapBev)
            .SetInferDataType(ge::InferDataType4BoxesOverlapBev);

        this->AICore().SetTiling(optiling::TilingFunc4BoxesOverlapBev);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910c");
    }
};

OP_ADD(BoxesOverlapBev);
} // namespace ops
