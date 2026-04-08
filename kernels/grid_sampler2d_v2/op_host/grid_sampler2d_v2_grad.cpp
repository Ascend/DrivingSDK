/*
* Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
*/
#include "ge/utils.h"
#include "grid_sampler2d_v2_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace std;

namespace {

constexpr int32_t FP32_BLOCK_NUM = 8;
constexpr int32_t BILINEAR_DIVIDE_UB_NUM = 47;
constexpr int32_t BUFFER_NUM = 2;

constexpr int32_t FLOAT_SIZE = 4;
constexpr int32_t ALIGN_256_BYTES = 256;
constexpr int32_t COORD_POSITION = 4;

constexpr int32_t CHANNEL_4 = 4;
constexpr int32_t CHANNEL_16 = 16;
constexpr int32_t CHANNEL_128 = 128;
constexpr int32_t FP32_GROUP_SIZE_LE_4 = 64;
constexpr int32_t FP32_GROUP_SIZE_GT_4_LE_16 = 32;
constexpr int32_t FP32_GROUP_SIZE_GT_16_LE_128 = 2;
} // namespace

namespace optiling {
static ge::graphStatus TilingFuncForGridSampler2dV2Grad(gert::TilingContext* context)
{
    GridSampler2dV2GradTilingData tilingData;
    // sys info
    CHECK_NULLPTR(context);
    auto platformInfoptr = context->GetPlatformInfo();
    CHECK_NULLPTR(platformInfoptr);
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    auto coreNum = ascendplatformInfo.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t availableUbSize;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, availableUbSize);

    // tensor input info
    auto gradShape = context->GetInputShape(0);
    auto xShape = context->GetInputShape(1);
    auto gridShape = context->GetInputShape(2);

    CHECK_NULLPTR(gradShape);
    CHECK_NULLPTR(xShape);
    CHECK_NULLPTR(gridShape);

    uint32_t batch = xShape->GetStorageShape().GetDim(0);
    uint32_t height = xShape->GetStorageShape().GetDim(1);
    uint32_t width = xShape->GetStorageShape().GetDim(2);
    uint32_t channel = xShape->GetStorageShape().GetDim(3);

    uint32_t gridH = gridShape->GetStorageShape().GetDim(1);
    uint32_t gridW = gridShape->GetStorageShape().GetDim(2);

    // attr input info
    auto attrsPtr = context->GetAttrs();
    CHECK_NULLPTR(attrsPtr);
    auto interpolationPtr = attrsPtr->GetAttrPointer<int64_t>(0);
    auto paddingPtr = attrsPtr->GetAttrPointer<int64_t>(1);
    auto alignCornersPtr = attrsPtr->GetAttrPointer<bool>(2);
    CHECK_NULLPTR(interpolationPtr);
    CHECK_NULLPTR(paddingPtr);
    CHECK_NULLPTR(alignCornersPtr);

    uint32_t interpolation = *interpolationPtr;
    uint32_t padding = *paddingPtr;
    bool alignCorners = *alignCornersPtr;

    // used core
    uint32_t usedCoreNum;
    uint32_t pNumPerCore;
    uint32_t tailPNum;
    uint64_t calcNum = batch * gridH * gridW;
    if (calcNum <= coreNum) {
        usedCoreNum = calcNum;
        pNumPerCore = 1;
        tailPNum = 0;
    } else {
        pNumPerCore = calcNum / coreNum;
        usedCoreNum = coreNum;
        tailPNum = calcNum % usedCoreNum;
    }

    int32_t groupSize = 1;
    if (channel <= CHANNEL_4) {
        groupSize = FP32_GROUP_SIZE_LE_4;
    } else if (channel <= CHANNEL_16) {
        groupSize = FP32_GROUP_SIZE_GT_4_LE_16;
    } else if (channel <= CHANNEL_128) {
        groupSize = FP32_GROUP_SIZE_GT_16_LE_128;
    }

    // allocate ub
    uint32_t alignedChannel;
    int32_t coordPosition = COORD_POSITION;
    alignedChannel = AlignUp(channel, FP32_BLOCK_NUM);
    uint32_t divideUbNum = BILINEAR_DIVIDE_UB_NUM;
    uint64_t extraUbSize = static_cast<uint64_t>(alignedChannel) * FLOAT_SIZE * coordPosition * groupSize * BUFFER_NUM;
    uint32_t calcCountPerLoop = AlignUp((availableUbSize - extraUbSize) / divideUbNum, ALIGN_256_BYTES) / FLOAT_SIZE;
    if (calcCountPerLoop <= 0) {
        return ge::GRAPH_FAILED;
    }

    // tiling setup
    tilingData.set_batch(batch);
    tilingData.set_pNumPerCore(pNumPerCore);
    tilingData.set_tailPNum(tailPNum);
    tilingData.set_channel(channel);
    tilingData.set_alignedChannel(alignedChannel);
    tilingData.set_height(height);
    tilingData.set_width(width);
    tilingData.set_gridH(gridH);
    tilingData.set_gridW(gridW);
    tilingData.set_blockNum(usedCoreNum);
    tilingData.set_calcCountPerLoop(calcCountPerLoop);
    tilingData.set_interpolation(interpolation);
    tilingData.set_padding(padding);
    tilingData.set_alignCorners(alignCorners);
    tilingData.set_groupSize(groupSize);
    tilingData.set_coordPosition(coordPosition);

    context->SetBlockDim(usedCoreNum);
    ADD_TILING_DATA(context, tilingData);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    CHECK_NULLPTR(currentWorkspace);
    currentWorkspace[0] = 16 * 1024 * 1024;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForGridSampler2dV2Grad(gert::InferShapeContext* context)
{
    CHECK_NULLPTR(context);
    const gert::Shape* gradShape = context->GetInputShape(0);
    const gert::Shape* xShape = context->GetInputShape(1);
    const gert::Shape* gridShape = context->GetInputShape(2);
    gert::Shape* dxShape = context->GetOutputShape(0);
    gert::Shape* dgridShape = context->GetOutputShape(1);

    CHECK_NULLPTR(gradShape);
    CHECK_NULLPTR(xShape);
    CHECK_NULLPTR(gridShape);
    CHECK_NULLPTR(dxShape);
    CHECK_NULLPTR(dgridShape);

    int32_t xN = xShape->GetDim(0);
    int32_t xH = xShape->GetDim(1);
    int32_t xW = xShape->GetDim(2);
    int32_t xC = xShape->GetDim(3);
    *dxShape = {xN, xH, xW, xC};

    int32_t gridN = gridShape->GetDim(0);
    int32_t gridH = gridShape->GetDim(1);
    int32_t gridW = gridShape->GetDim(2);
    *dgridShape = {gridN, gridH, gridW, 2};

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForGridSampler2dV2Grad(gert::InferDataTypeContext* context)
{
    CHECK_NULLPTR(context);
    const ge::DataType xDtype = context->GetInputDataType(1);
    const ge::DataType gridDtype = context->GetInputDataType(2);
    context->SetOutputDataType(0, xDtype);
    context->SetOutputDataType(1, gridDtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class GridSampler2dV2Grad : public OpDef {
public:
    explicit GridSampler2dV2Grad(const char* name) : OpDef(name)
    {
        this->Input("grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({{ge::FORMAT_ND}})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({{ge::FORMAT_ND}})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("grid")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({{ge::FORMAT_ND}})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({{ge::FORMAT_ND}})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dgrid")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({{ge::FORMAT_ND}})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("interpolation_mode").AttrType(REQUIRED).Int();
        this->Attr("padding_mode").AttrType(REQUIRED).Int();
        this->Attr("align_corners").AttrType(REQUIRED).Bool();
        this->SetInferShape(ge::InferShapeForGridSampler2dV2Grad)
            .SetInferDataType(ge::InferDataTypeForGridSampler2dV2Grad);
        this->AICore().SetTiling(optiling::TilingFuncForGridSampler2dV2Grad);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(GridSampler2dV2Grad);
} // namespace ops