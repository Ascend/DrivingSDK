#include "boxes_overlap_bev_v1_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "csrc/utils.h"
namespace {
constexpr float AVALIABLE_UB_RATIO = 0.6;
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

const uint32_t TILE_N_910B = 64;
}   // some const express

namespace optiling {
static ge::graphStatus TilingFunc4BoxesOverlapBevV1(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    /* get some global information: aivNum, ubSize */
    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto aivNum = ascendplatformInfo.GetCoreNumAiv();
    
    context->SetBlockDim(aivNum);

    uint64_t ubSize;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize *= AVALIABLE_UB_RATIO;

    if (aivNum == 0) {
        return ge::GRAPH_FAILED;
    }

    /* Compute tiling information */
    auto boxesATensorPtr = context->GetInputTensor(POS_INPUT_BOXES_A);
    auto boxesBTensorPtr = context->GetInputTensor(POS_INPUT_BOXES_B);
    auto attrs = context->GetAttrs();
    if (boxesATensorPtr == nullptr || boxesBTensorPtr == nullptr || attrs == nullptr) {
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

    uint32_t M = boxesANum;
    uint32_t N = boxesBNum;
    uint32_t tileN = TILE_N_910B;
    uint32_t tileCountM = M;
    uint32_t tileCountN = Ceil(N, tileN);
    
    /* Set tilingData */
    BoxesOverlapBevV1TilingData tilingData;
    
    tilingData.set_M(M);
    tilingData.set_N(N);
    tilingData.set_totalCoreCount(aivNum);
    tilingData.set_tileCountN(tileCountN);
    tilingData.set_tileCountM(tileCountM);
    tilingData.set_tileN(tileN);
    tilingData.set_modeFlag(modeFlag);
    tilingData.set_margin(margin);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t systemWorkspaceSize = ascendplatformInfo.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus Infershape4BoxesOverlapBevV1(gert::InferShapeContext *context)
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

static ge::graphStatus InferDataType4BoxesOverlapBevV1(gert::InferDataTypeContext *context)
{
    const ge::DataType box_dtype = context->GetInputDataType(POS_INPUT_BOXES_A);
    context->SetOutputDataType(POS_OUTPUT_RES, box_dtype);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class BoxesOverlapBevV1 : public OpDef {
public:
    explicit BoxesOverlapBevV1(const char* name) : OpDef(name)
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

        this->SetInferShape(ge::Infershape4BoxesOverlapBevV1)
            .SetInferDataType(ge::InferDataType4BoxesOverlapBevV1);

        this->AICore().SetTiling(optiling::TilingFunc4BoxesOverlapBevV1);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(BoxesOverlapBevV1);
}