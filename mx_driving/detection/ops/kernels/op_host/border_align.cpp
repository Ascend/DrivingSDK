/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "border_align_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cmath>

using namespace std;
namespace optiling {
const uint32_t TILE_NUM = 8;
const uint32_t BOX_INFO = 4; // 每个box有四个坐标
static ge::graphStatus TilingForBorderAlign(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    BorderAlignTilingData tiling;

    if (context->GetInputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputShape(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    auto inputShape = context->GetInputShape(0)->GetStorageShape(); // [B, H, W, C]
    auto roisShape = context->GetInputShape(1)->GetStorageShape(); // [B, H * W, 4]
    auto outputShape = context->GetOutputShape(0)->GetStorageShape(); // [B, H * W, P + 1, C]

    uint32_t batchSize = inputShape.GetDim(0);
    uint32_t inputH = inputShape.GetDim(1);
    uint32_t inputW = inputShape.GetDim(2);
    uint32_t channels = inputShape.GetDim(3);
    // channels必须要被4整除
    if (channels % BOX_INFO != 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t channelsAligned;
    if (static_cast<uint32_t>((channels / BOX_INFO) % TILE_NUM) == 0) {
        channelsAligned = channels;
    } else {
        channelsAligned = (static_cast<uint32_t>(channels / BOX_INFO / TILE_NUM) + 1) * TILE_NUM * BOX_INFO;
    }

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int32_t pooledSize = *(attrsPtr->GetAttrPointer<int32_t>(0));

    uint32_t roisNum = roisShape.GetDim(0) * roisShape.GetDim(1);
    if (roisNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto platform = context->GetPlatformInfo();
    if (platform == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    auto platform_info = platform_ascendc::PlatformAscendC(platform);
    uint32_t BLOCK_DIM = platform_info.GetCoreNumAiv();
    if (BLOCK_DIM == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t roisNumAligned;
    if (static_cast<uint32_t>(roisNum % TILE_NUM) == 0) {
        roisNumAligned = roisNum;
    } else {
        roisNumAligned = (static_cast<uint32_t>(roisNum / TILE_NUM) + 1) * TILE_NUM;
    }

    uint32_t tailNum = roisNumAligned - roisNum;
    uint32_t roisNumPerScore = (roisNumAligned / BLOCK_DIM / TILE_NUM) * TILE_NUM;
    uint32_t roisNumPerLcore = roisNumPerScore + TILE_NUM;
    uint32_t scoreNum = (BLOCK_DIM * (TILE_NUM + roisNumPerScore) - roisNumAligned) / TILE_NUM;
    uint32_t lcoreNum = BLOCK_DIM - scoreNum;
    
    if (roisNumPerScore == 0) {
        BLOCK_DIM = BLOCK_DIM - scoreNum;
    }
    if (roisNumPerLcore == 0) {
        BLOCK_DIM = BLOCK_DIM - lcoreNum;
    }

    uint32_t inputBufferSize = channelsAligned / BOX_INFO * sizeof(float);
    // 每次取64个RoI进UB，目前是写死的，以后做性能优化可以考虑一下UB的内存空间计算最合理的值
    uint32_t roisNumPerLoop = 64;
    uint32_t roisBufferSize = roisNumPerLoop * BOX_INFO * sizeof(float);
    uint32_t moveInLength = channelsAligned / BOX_INFO;
    uint32_t moveOutLength = channels / BOX_INFO * sizeof(float);

    tiling.set_roisNumPerLoop(roisNumPerLoop);
    tiling.set_batchSize(batchSize);
    tiling.set_inputH(inputH);
    tiling.set_inputW(inputW);
    tiling.set_channels(channels);
    tiling.set_moveInLength(moveInLength);
    tiling.set_moveOutLength(moveOutLength);
    tiling.set_roisNumAligned(roisNumAligned);
    tiling.set_tailNum(tailNum);
    tiling.set_pooledSize(pooledSize);
    tiling.set_roisNumPerLcore(roisNumPerLcore);
    tiling.set_roisNumPerScore(roisNumPerScore);
    tiling.set_lcoreNum(lcoreNum);
    tiling.set_scoreNum(scoreNum);
    tiling.set_inputBufferSize(inputBufferSize);
    tiling.set_roisBufferSize(roisBufferSize);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(BLOCK_DIM);

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* roisShape = context->GetInputShape(1);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (roisShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t batchSize = inputShape->GetDim(0);
    int64_t heightTimesWidth = roisShape->GetDim(1);
    int64_t channels = inputShape->GetDim(3);
    auto attrsPtr = context->GetAttrs();
    uint32_t pooledSize = *(attrsPtr->GetAttrPointer<uint32_t>(0));
    
    *outputShape = {batchSize, heightTimesWidth, pooledSize + 1, channels};

    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeBorderAlign(gert::InferDataTypeContext* context)
{
    const ge::DataType valueDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, valueDtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class BorderAlign : public OpDef {
public:
    explicit BorderAlign(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("rois")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("pooledSize").AttrType(REQUIRED).Int();
        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataTypeBorderAlign);
        this->AICore()
            .SetTiling(optiling::TilingForBorderAlign);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(BorderAlign);
}