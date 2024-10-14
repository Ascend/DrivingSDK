/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */

#include "hypot_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
constexpr uint32_t SIZE_OF_FLOAT = 4;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t ALIGN_NUM = BLOCK_SIZE / SIZE_OF_FLOAT;
constexpr uint32_t RESERVED_UB_SIZE = 16 * 1024;
constexpr uint32_t BUFFER_NUM = 2;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto coreNum  = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatform;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    context->SetBlockDim(coreNum);

    uint32_t totalLengthAligned = ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    uint32_t formerNum = (totalLengthAligned / ALIGN_NUM) % coreNum;
    uint32_t tailNum = coreNum - formerNum;

    uint32_t baseLength = totalLengthAligned / ALIGN_NUM / coreNum;
    uint32_t formerLength = (baseLength + (formerNum ? 1 : 0)) * ALIGN_NUM;
    uint32_t tailLength = baseLength * ALIGN_NUM;

    // there are 3 queues in pipe
    uint32_t formerSlice = formerLength * 3 * SIZE_OF_FLOAT * BUFFER_NUM / (ubSizePlatform - RESERVED_UB_SIZE) + 1;
    uint32_t tailSlice = tailLength * 3 * SIZE_OF_FLOAT * BUFFER_NUM / (ubSizePlatform - RESERVED_UB_SIZE) + 1;

    uint32_t formerTileLength = (formerLength / formerSlice + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
    uint32_t tailTileLength = (tailLength / tailSlice + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
    uint32_t formerTileNum = (formerTileLength == 0) ? 0 : formerLength / formerTileLength + ((formerLength % formerTileLength) ? 1 : 0);
    uint32_t tailTileNum = (tailTileLength == 0) ? 0 : tailLength / tailTileLength + ((tailLength % tailTileLength) ? 1 : 0);
    uint32_t formerRemainTileLength = (formerTileNum == 0) ? 0 : formerLength - (formerTileNum - 1) * formerTileLength;
    uint32_t tailRemainTileLength = (tailTileNum == 0) ? 0 : tailLength - (tailTileNum - 1) * tailTileLength;

    HypotTilingData tiling;
    tiling.set_formerNum(formerNum);
    tiling.set_tailNum(tailNum);
    tiling.set_formerLength(formerLength);
    tiling.set_tailLength(tailLength);
    tiling.set_formerTileLength(formerTileLength);
    tiling.set_tailTileLength(tailTileLength);
    tiling.set_formerTileNum(formerTileNum);
    tiling.set_tailTileNum(tailTileNum);
    tiling.set_formerRemainTileLength(formerRemainTileLength);
    tiling.set_tailRemainTileLength(tailRemainTileLength);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus Infershape(gert::InferShapeContext *context)
{
    const auto inputShape = context->GetInputShape(0);
    auto outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype(gert::InferDataTypeContext *context)
{
    const auto out_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, out_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Hypot : public OpDef {
public:
    explicit Hypot(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("other")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::Infershape)
            .SetInferDataType(ge::InferDtype);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(Hypot);
}
