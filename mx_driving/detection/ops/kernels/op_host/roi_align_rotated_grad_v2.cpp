/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "roi_align_rotated_grad_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
using namespace ge;
using namespace std;

namespace {
const uint32_t INPUT_INPUT = 0;
const uint32_t INPUT_ROIS = 1;
const uint32_t INPUT_GRAD_OUTPUT = 2;

const uint32_t INPUT_POOLED_HEIGHT = 0;
const uint32_t INPUT_POOLED_WIDTH = 1;
const uint32_t INPUT_SPATIAL_SCALE = 2;
const uint32_t INPUT_SAMPLING_RATIO = 3;
const uint32_t INPUT_ALIGNED = 4;
const uint32_t INPUT_CLOCKWISE = 5;

const uint32_t BOX_SIZE_DIM = 1;
const uint32_t BATCH_SIZE_DIM = 0;
const uint32_t CHANNEL_DIM = 1;
const uint32_t HEIGHT_DIM = 2;
const uint32_t WIDTH_DIM = 3;

const uint32_t OUTPUT_GRAD_INPUT = 0;

const uint32_t WORKSAPCE_16MBYTE_SIZE = 16 * 1024 * 1024;
} // namespace

namespace optiling {
static ge::graphStatus TilingFuncForRoiAlignRotatedGradV2(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    RoiAlignRotatedGradV2TilingData tiling;
    auto inputTensorPtr = context->GetInputTensor(INPUT_INPUT);
    auto RoiTensorPtr = context->GetInputTensor(INPUT_ROIS);
    auto gradOutputTensorPtr = context->GetInputTensor(INPUT_GRAD_OUTPUT);
    if (inputTensorPtr == nullptr || RoiTensorPtr == nullptr || gradOutputTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto inputShape = inputTensorPtr->GetStorageShape();
    auto RoiShape = RoiTensorPtr->GetStorageShape();

    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    uint32_t coreNum = ascendplatformInfo.GetCoreNumAiv();
    context->SetBlockDim(coreNum);
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t boxSize = RoiShape.GetDim(BOX_SIZE_DIM);
    int32_t pooledHeight = *(attrs->GetAttrPointer<uint32_t>(INPUT_POOLED_HEIGHT));
    int32_t pooledWidth = *(attrs->GetAttrPointer<uint32_t>(INPUT_POOLED_WIDTH));

    uint32_t batchSize = inputShape.GetDim(BATCH_SIZE_DIM);
    uint32_t channelNum = inputShape.GetDim(CHANNEL_DIM);
    uint32_t width = inputShape.GetDim(WIDTH_DIM);
    uint32_t height = inputShape.GetDim(HEIGHT_DIM);

    bool aligned = *(attrs->GetAttrPointer<bool>(INPUT_ALIGNED));
    bool clockwise = *(attrs->GetAttrPointer<bool>(INPUT_CLOCKWISE));
    int32_t samplingRatio = *(attrs->GetAttrPointer<uint32_t>(INPUT_SAMPLING_RATIO));
    float spatialScale = *(attrs->GetAttrPointer<float>(INPUT_SPATIAL_SCALE));

    uint32_t coreRoisNums = boxSize / coreNum;
    uint32_t coreRoisTail = boxSize % coreNum;

    tiling.set_coreRoisNums(coreRoisNums);
    tiling.set_coreRoisTail(coreRoisTail);
    tiling.set_boxSize(boxSize);
    tiling.set_pooledHeight(pooledHeight);
    tiling.set_pooledWidth(pooledWidth);
    tiling.set_batchSize(batchSize);
    tiling.set_channelNum(channelNum);
    tiling.set_width(width);
    tiling.set_height(height);
    tiling.set_aligned(aligned);
    tiling.set_clockwise(clockwise);
    tiling.set_samplingRatio(samplingRatio);
    tiling.set_spatialScale(spatialScale);
    tiling.set_coreNum(coreNum);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = WORKSAPCE_16MBYTE_SIZE;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShapeForRoiAlignRotatedGradV2(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(INPUT_INPUT);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* gradInputShape = context->GetOutputShape(OUTPUT_GRAD_INPUT);
    if (gradInputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gradInputShape->AppendDim(inputShape->GetDim(BATCH_SIZE_DIM));
    gradInputShape->AppendDim(inputShape->GetDim(HEIGHT_DIM));
    gradInputShape->AppendDim(inputShape->GetDim(WIDTH_DIM));
    gradInputShape->AppendDim(inputShape->GetDim(CHANNEL_DIM));
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeForRoiAlignRotatedGradV2(gert::InferDataTypeContext* context)
{
    auto inputDtype = context->GetInputDataType(INPUT_INPUT);
    context->SetOutputDataType(OUTPUT_GRAD_INPUT, inputDtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class RoiAlignRotatedGradV2 : public OpDef {
public:
    explicit RoiAlignRotatedGradV2(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("rois")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("grad_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("pooled_height").AttrType(REQUIRED).Int();
        this->Attr("pooled_width").AttrType(REQUIRED).Int();
        this->Attr("spatial_scale").AttrType(REQUIRED).Float();
        this->Attr("sampling_ratio").AttrType(REQUIRED).Int();
        this->Attr("aligned").AttrType(REQUIRED).Bool();
        this->Attr("clockwise").AttrType(REQUIRED).Bool();
        this->Output("grad_input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForRoiAlignRotatedGradV2)
            .SetInferDataType(ge::InferDataTypeForRoiAlignRotatedGradV2);
        this->AICore().SetTiling(optiling::TilingFuncForRoiAlignRotatedGradV2);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(RoiAlignRotatedGradV2);
}