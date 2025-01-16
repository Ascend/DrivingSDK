/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#include "csrc/utils.h"
#include "grid_sampler2d_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t INTERPOLATION_INDEX = 0;
constexpr size_t PADDING_INDEX = 1;
constexpr size_t ALIGN_INDEX = 2;
constexpr size_t DIM_INDEX_0 = 0;
constexpr size_t DIM_INDEX_1 = 1;
constexpr size_t DIM_INDEX_2 = 2;
constexpr size_t DIM_INDEX_3 = 3;
constexpr size_t X_INPUT_INDEX = 0;
constexpr size_t GRID_INPUT_INDEX = 1;
constexpr size_t Y_OUTPUT_INDEX = 0;

constexpr int32_t B32_DATA_NUM_PER_BLOCK = 8;
constexpr int32_t B32_BYTE_SIZE = 4;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t RESERVE_UB = 10 * 1024; // 10 KB
constexpr int32_t BILINEAR_DIVIDE_UB_NUM = 40;
constexpr int32_t COORD_POSITION = 4;
constexpr int32_t CHANNEL_64 = 64;
constexpr int32_t CHANNEL_128 = 128;
constexpr int32_t FP32_GROUP_SIZE_LE_64 = 32;
constexpr int32_t FP32_GROUP_SIZE_GT_64_LE_128 = 16;
} // namespace

namespace optiling {
static ge::graphStatus TilingFuncForGridSampler2dV2(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int32_t coreNum = ascendPlatformInfo.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto interpolationModePtr = attrsPtr->GetAttrPointer<int64_t>(INTERPOLATION_INDEX);
    auto paddingModePtr = attrsPtr->GetAttrPointer<int64_t>(PADDING_INDEX);
    auto alignCornersPtr = attrsPtr->GetAttrPointer<bool>(ALIGN_INDEX);
    if ((interpolationModePtr == nullptr) || (paddingModePtr == nullptr) || (alignCornersPtr == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    int64_t interpolationMode = *interpolationModePtr;
    int64_t paddingMode = *paddingModePtr;
    bool alignCorners = *alignCornersPtr;

    auto xTransShapePtr = context->GetInputShape(X_INPUT_INDEX);
    auto gridShapePtr = context->GetInputShape(GRID_INPUT_INDEX);
    if ((xTransShapePtr == nullptr) || (gridShapePtr == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    auto xTransShape = xTransShapePtr->GetStorageShape(); // n, hIn, wIn, c
    auto gridShape = gridShapePtr->GetStorageShape();     // n, hOut, wOut, c
    int32_t batchSize = xTransShape.GetDim(DIM_INDEX_0);
    int32_t inHeight = xTransShape.GetDim(DIM_INDEX_1);
    int32_t inWidth = xTransShape.GetDim(DIM_INDEX_2);
    int32_t channel = xTransShape.GetDim(DIM_INDEX_3);
    int32_t outHeight = gridShape.GetDim(DIM_INDEX_1);
    int32_t outWidth = gridShape.GetDim(DIM_INDEX_2);

    int32_t totalTaskNum = batchSize * outHeight * outWidth;
    if (totalTaskNum == 0) {
        return ge::GRAPH_FAILED;
    }

    int32_t taskNumPerCore = Ceil(totalTaskNum, coreNum);
    if (taskNumPerCore == 0) {
        return ge::GRAPH_FAILED;
    }
    int32_t taskNumRemained = (totalTaskNum % taskNumPerCore == 0) ? taskNumPerCore : (totalTaskNum % taskNumPerCore);
    int32_t usedCoreNum = Ceil(totalTaskNum, taskNumPerCore);
    context->SetBlockDim(usedCoreNum);

    int32_t alignedChannel = AlignUp(channel, B32_DATA_NUM_PER_BLOCK);

    uint64_t availableUbBytes;
    ascendPlatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, availableUbBytes);
    int32_t divideNum = BILINEAR_DIVIDE_UB_NUM;
    int32_t coordPosition = COORD_POSITION;

    int32_t groupSize = 1;
    if (channel <= CHANNEL_64) {
        groupSize = FP32_GROUP_SIZE_LE_64;
    } else if (channel <= CHANNEL_128) {
        groupSize = FP32_GROUP_SIZE_GT_64_LE_128;
    }

    uint64_t extraUbBytes =
        static_cast<uint64_t>(alignedChannel) * B32_BYTE_SIZE * coordPosition * groupSize * (BUFFER_NUM * 2 + 1);
    int32_t taskNumPerLoop = (availableUbBytes - RESERVE_UB - extraUbBytes) / (divideNum * B32_BYTE_SIZE);
    if (taskNumPerLoop <= 0) {
        return ge::GRAPH_FAILED;
    }
    int32_t alignedTaskNumPerLoop = taskNumPerLoop / B32_DATA_NUM_PER_BLOCK * B32_DATA_NUM_PER_BLOCK;

    GridSampler2dV2TilingData tilingData;
    tilingData.set_interpolationMode(interpolationMode);
    tilingData.set_paddingMode(paddingMode);
    tilingData.set_alignCorners(alignCorners);
    tilingData.set_batchSize(batchSize);
    tilingData.set_channel(channel);
    tilingData.set_inHeight(inHeight);
    tilingData.set_inWidth(inWidth);
    tilingData.set_outHeight(outHeight);
    tilingData.set_outWidth(outWidth);
    tilingData.set_taskNumPerCore(taskNumPerCore);
    tilingData.set_usedCoreNum(usedCoreNum);
    tilingData.set_alignedChannel(alignedChannel);
    tilingData.set_alignedTaskNumPerLoop(alignedTaskNumPerLoop);
    tilingData.set_copyLoop(taskNumPerCore / alignedTaskNumPerLoop);
    tilingData.set_copyTail(taskNumPerCore % alignedTaskNumPerLoop);
    tilingData.set_lastCopyLoop(taskNumRemained / alignedTaskNumPerLoop);
    tilingData.set_lastCopyTail(taskNumRemained % alignedTaskNumPerLoop);
    tilingData.set_coordPosition(coordPosition);
    tilingData.set_groupSize(groupSize);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 16 * 1024 * 1024;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForGridSampler2dV2(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(X_INPUT_INDEX);
    const gert::Shape* gridShape = context->GetInputShape(GRID_INPUT_INDEX);
    gert::Shape* yShape = context->GetOutputShape(Y_OUTPUT_INDEX);
    if (xShape == nullptr || gridShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int32_t n = xShape->GetDim(DIM_INDEX_0);
    int32_t c = xShape->GetDim(DIM_INDEX_3);
    int32_t cOut = AlignUp(c, B32_DATA_NUM_PER_BLOCK);
    int32_t hOut = gridShape->GetDim(DIM_INDEX_1);
    int32_t wOut = gridShape->GetDim(DIM_INDEX_2);
    *yShape = {n, hOut, wOut, cOut};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForGridSampler2dV2(gert::InferDataTypeContext* context)
{
    const ge::DataType valueDtype = context->GetInputDataType(X_INPUT_INDEX);
    context->SetOutputDataType(Y_OUTPUT_INDEX, valueDtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class GridSampler2dV2 : public OpDef {
public:
    explicit GridSampler2dV2(const char* name) : OpDef(name)
    {
        this->Input("x_trans")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("grid")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y_trans")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("interpolation_mode").AttrType(REQUIRED).Int();
        this->Attr("padding_mode").AttrType(REQUIRED).Int();
        this->Attr("align_corners").AttrType(REQUIRED).Bool();

        this->SetInferShape(ge::InferShapeForGridSampler2dV2).SetInferDataType(ge::InferDataTypeForGridSampler2dV2);
        this->AICore().SetTiling(optiling::TilingFuncForGridSampler2dV2);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GridSampler2dV2);
} // namespace ops