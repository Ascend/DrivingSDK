/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

#include "fused_bias_leaky_relu_v2_tiling.h"

using namespace ge;
using namespace std;
using namespace AscendC;

namespace {
constexpr size_t SLOPE_IDX = 0;
constexpr size_t SCALE_IDX = 1;
}

namespace optiling {
constexpr uint32_t BUFFER_NUM = 2;

constexpr uint32_t SIZE_OF_DATA = 4;
constexpr uint32_t BLOCK_SIZE = 32 * 1024;
constexpr int32_t SINGLE_BLOCK = BLOCK_SIZE / SIZE_OF_DATA;


static ge::graphStatus TilingForFusedBiasLeakyReluV2(gert::TilingContext* context)
{
    FusedBiasLeakyReluV2TilingData tiling;

    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t numbersOfCore = ascendPlatformInfo.GetCoreNumAiv();
    if (numbersOfCore == 0) {
        return ge::GRAPH_FAILED;
    }

    auto dtype = context->GetInputDesc(0)->GetDataType();
    int32_t singleBlock = SINGLE_BLOCK;
    if (ge::DT_FLOAT16 == dtype) {
        singleBlock = BLOCK_SIZE / 2;
    }

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    float negative_slope = *(attrsPtr->GetAttrPointer<float>(SLOPE_IDX));
    float scale = *(attrsPtr->GetAttrPointer<float>(SCALE_IDX));

    auto inputTensorPtr = context->GetInputTensor(0);
    if (inputTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t totalDataLength = inputTensorPtr->GetShapeSize();
    uint32_t totalLengthAligned = ((totalDataLength + singleBlock - 1) / singleBlock) * singleBlock;

    uint32_t totalTask = ((totalLengthAligned + singleBlock - 1) / singleBlock);
    uint32_t average = totalTask / numbersOfCore;
    uint32_t remainder = totalTask % numbersOfCore;
    uint32_t usedCoreNum = numbersOfCore;
    if (average == 0) {
        usedCoreNum = remainder;
    }

    context->SetBlockDim(usedCoreNum);

    tiling.set_negative_slope(negative_slope);
    tiling.set_scale(scale);

    tiling.set_average(average);
    tiling.set_remainder(remainder);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_totalDataLength(totalDataLength);
    tiling.set_singleBlock(singleBlock);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShapeForFusedBiasLeakyReluV2(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    if (xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForFusedBiasLeakyReluV2(gert::InferDataTypeContext* context)
{
    const ge::DataType value_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, value_dtype);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class FusedBiasLeakyReluV2 : public OpDef {
public:
    explicit FusedBiasLeakyReluV2(const char* name) : OpDef(name)
    {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("bias")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("output")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("negative_slope").Float();
    this->Attr("scale").Float();

    this->SetInferShape(ge::InferShapeForFusedBiasLeakyReluV2)
        .SetInferDataType(ge::InferDataTypeForFusedBiasLeakyReluV2);;

    this->AICore().SetTiling(optiling::TilingForFusedBiasLeakyReluV2);
    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(FusedBiasLeakyReluV2);
}