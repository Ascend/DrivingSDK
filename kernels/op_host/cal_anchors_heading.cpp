#include "cal_anchors_heading_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "csrc/utils.h"
namespace {
    constexpr float AVALIABLE_UB_RATIO = 0.5;
    constexpr uint32_t DATA_BLOCK_SIZE = 32;
    constexpr uint32_t ELEM_BYTE_SIZE = sizeof(float);
    constexpr uint32_t ONE_REPEAT_BYTE_SIZE = 256;
    constexpr uint32_t MAX_REPEAT_TIMES = 255;
    constexpr uint32_t ANCHORS_INPUT_IDX = 0;
    constexpr uint32_t ORIGIN_POS_INPUT_IDX = 1;
    constexpr uint32_t HEADING_OUTPUT_IDX = 0;
    constexpr uint32_t BATCH_SIZE_IDX = 0;
    constexpr uint32_t ANCHORS_NUM_IDX = 1;
    constexpr uint32_t SEQ_LENGTH_IDX = 2;
    constexpr uint32_t UB_TASK_BUFFER_COUNT = 6;
    constexpr uint32_t TASK_SEQ_COUNT = 2;
}   // some const express

namespace optiling {
static ge::graphStatus TilingForCalAnchorsHeading(gert::TilingContext* context)
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
    auto dTypePtr = context->GetInputDesc(ANCHORS_INPUT_IDX);
    auto anchorsShapePtr = context->GetInputShape(ANCHORS_INPUT_IDX);
    auto originPosShapePtr = context->GetInputShape(ORIGIN_POS_INPUT_IDX);
    auto headingShapePtr = context->GetOutputShape(HEADING_OUTPUT_IDX);
    if (dTypePtr == nullptr || anchorsShapePtr == nullptr ||
        originPosShapePtr == nullptr || headingShapePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto anchorsShape = anchorsShapePtr->GetStorageShape();
    uint32_t batchSize = anchorsShape.GetDim(BATCH_SIZE_IDX);
    uint32_t anchorsNum = anchorsShape.GetDim(ANCHORS_NUM_IDX);
    uint32_t seqLength = anchorsShape.GetDim(SEQ_LENGTH_IDX);
    uint32_t taskLength = seqLength * TASK_SEQ_COUNT;

    uint32_t coreAnchorNumTask = Ceil(batchSize * anchorsNum, aivNum);
    uint32_t bigCoreCount = (batchSize * anchorsNum) % aivNum;
    if (bigCoreCount == 0) {
        bigCoreCount = aivNum;
    }
    uint32_t taskDataBlockCount = Ceil(seqLength * ELEM_BYTE_SIZE, DATA_BLOCK_SIZE);
    uint32_t taskElemCountAligned = taskDataBlockCount * (DATA_BLOCK_SIZE / ELEM_BYTE_SIZE);
    uint32_t taskMemAlignedByte = taskDataBlockCount * DATA_BLOCK_SIZE;
    
    uint16_t copyInLocalStride = Ceil(taskLength * ELEM_BYTE_SIZE, DATA_BLOCK_SIZE) % TASK_SEQ_COUNT;
    uint32_t copyInDataBlockElemCountAligned = (Ceil(taskLength * ELEM_BYTE_SIZE, DATA_BLOCK_SIZE) +
        copyInLocalStride) * (DATA_BLOCK_SIZE / ELEM_BYTE_SIZE);

    uint32_t singleLoopTask = ubSize / (taskMemAlignedByte * UB_TASK_BUFFER_COUNT);
    singleLoopTask = std::min(std::min(singleLoopTask, (MAX_REPEAT_TIMES * ONE_REPEAT_BYTE_SIZE) /
        taskMemAlignedByte), MAX_REPEAT_TIMES);

    /* Set tilingData */
    CalAnchorsHeadingTilingData tilingData;
    tilingData.set_batchSize(batchSize);
    tilingData.set_anchorsNum(anchorsNum);
    tilingData.set_seqLength(seqLength);
    tilingData.set_coreAnchorNumTask(coreAnchorNumTask);
    tilingData.set_taskMemAlignedByte(taskMemAlignedByte);
    tilingData.set_taskElemCountAligned(taskElemCountAligned);
    tilingData.set_bigCoreCount(bigCoreCount);
    tilingData.set_singleLoopTask(singleLoopTask);
    tilingData.set_copyInLocalStride(copyInLocalStride);
    tilingData.set_copyInDataBlockElemCountAligned(copyInDataBlockElemCountAligned);

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
static ge::graphStatus InferShapeForCalAnchorsHeading(gert::InferShapeContext* context)
{
    const gert::Shape* anchors = context->GetInputShape(ANCHORS_INPUT_IDX);
    const gert::Shape* originPos = context->GetInputShape(ORIGIN_POS_INPUT_IDX);
    gert::Shape* heading = context->GetOutputShape(HEADING_OUTPUT_IDX);
    if (anchors == nullptr || originPos == nullptr || heading == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t batchSize = anchors->GetDim(BATCH_SIZE_IDX);
    int64_t anchorNums = anchors->GetDim(ANCHORS_NUM_IDX);
    int64_t seqLength = anchors->GetDim(SEQ_LENGTH_IDX);
    *heading = {batchSize, anchorNums, seqLength};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForCalAnchorsHeading(gert::InferDataTypeContext* context)
{
    const ge::DataType value_dtype = context->GetInputDataType(ANCHORS_INPUT_IDX);
    context->SetOutputDataType(HEADING_OUTPUT_IDX, value_dtype);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class CalAnchorsHeading : public OpDef {
public:
    explicit CalAnchorsHeading(const char* name) : OpDef(name)
    {
        this->Input("anchors")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("origin_pos")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("heading")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForCalAnchorsHeading).SetInferDataType(ge::InferDataTypeForCalAnchorsHeading);
        this->AICore().SetTiling(optiling::TilingForCalAnchorsHeading);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(CalAnchorsHeading);
}
