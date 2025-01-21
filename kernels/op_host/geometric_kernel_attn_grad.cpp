/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
 */
#include "geometric_kernel_attn_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "ge/utils.h"

using namespace ge;
using namespace std;

namespace {
const uint32_t POS_INPUT_VALUE = 0;
const uint32_t POS_INPUT_SPATIAL_SHAPES = 1;
const uint32_t POS_INPUT_LEVEL_START_INDEX = 2;
const uint32_t POS_INPUT_SAMPLING_LOCATIONS = 3;
const uint32_t POS_INPUT_ATTN_WEIGHTS = 4;
const uint32_t POS_INPUT_GRAD_OUTPUT = 5;

const uint32_t POS_OUTPUT_GRAD_VALUE = 0;
const uint32_t POS_OUTPUT_GRAD_ATTN_WEIGHTS = 1;

const uint32_t VALUE_BATCH_SIZE_DIM = 0;
const uint32_t VALUE_NUM_KEYS_DIM = 1;
const uint32_t VALUE_NUM_HEADS_DIM = 2;
const uint32_t VALUE_EMBED_DIMS_DIM = 3;
const uint32_t ATTN_WEIGHTS_BATCH_SIZE_DIM = 0;
const uint32_t ATTN_WEIGHTS_NUM_QUERIES_DIM = 1;
const uint32_t ATTN_WEIGHTS_NUM_LEVELS_DIM = 3;
const uint32_t ATTN_WEIGHTS_NUM_POINTS_DIM = 4;

const uint64_t UB_RESERVE_BYTES = 10 * 1024;
const uint32_t FLOAT32_BYTES = 4;
const uint32_t BLOCK_BYTES = 32;
} // namespace

namespace optiling {
static ge::graphStatus TilingFuncForGeometricKernelAttnGrad(gert::TilingContext* context)
{
    GeometricKernelAttnGradTilingData tiling;

    auto valueTensorPtr = context->GetInputTensor(POS_INPUT_VALUE);
    auto attnWeightsTensorPtr = context->GetInputTensor(POS_INPUT_ATTN_WEIGHTS);
    if (valueTensorPtr == nullptr || attnWeightsTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto valueShape = valueTensorPtr->GetStorageShape();
    auto attnWeightShape = attnWeightsTensorPtr->GetStorageShape();

    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t coreNum = ascendPlatformInfo.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(coreNum);

    uint32_t batchSize = valueShape.GetDim(VALUE_BATCH_SIZE_DIM);
    uint32_t numHeads = valueShape.GetDim(VALUE_NUM_HEADS_DIM);
    uint32_t embedDims = valueShape.GetDim(VALUE_EMBED_DIMS_DIM);
    uint32_t numKeys = valueShape.GetDim(VALUE_NUM_KEYS_DIM);
    uint32_t numLevels = attnWeightShape.GetDim(ATTN_WEIGHTS_NUM_LEVELS_DIM);
    uint32_t numQueries = attnWeightShape.GetDim(ATTN_WEIGHTS_NUM_QUERIES_DIM);
    uint32_t numPoints = attnWeightShape.GetDim(ATTN_WEIGHTS_NUM_POINTS_DIM);

    uint32_t numItemsPerBlock = BLOCK_BYTES / FLOAT32_BYTES;
    uint32_t numLevelsAligned = AlignUp(numLevels, numItemsPerBlock);
    uint32_t numPointsAligned = AlignUp(numPoints, numItemsPerBlock);

    uint32_t taskNum = batchSize * numQueries;
    uint32_t taskPerCore = taskNum / coreNum;
    uint32_t taskCoreTail = taskNum % coreNum;

    uint64_t ubBytesTotal;
    ascendPlatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytesTotal);
    uint64_t ubBytes = ubBytesTotal - UB_RESERVE_BYTES;
    uint64_t ubSize = ubBytes / FLOAT32_BYTES;
    uint32_t ubSize4Others = 5 * numLevelsAligned + 3 * numHeads * numLevels * numPointsAligned * embedDims;
    uint32_t querySizePerTask = 10 * numHeads * numLevels * numPointsAligned + numHeads * embedDims;
    uint32_t taskCompNum = (ubSize - ubSize4Others - numItemsPerBlock) / querySizePerTask;

    tiling.set_batchSize(batchSize);
    tiling.set_numHeads(numHeads);
    tiling.set_embedDims(embedDims);
    tiling.set_numKeys(numKeys);
    tiling.set_numLevels(numLevels);
    tiling.set_numQueries(numQueries);
    tiling.set_numPoints(numPoints);
    tiling.set_coreNum(coreNum);
    tiling.set_taskPerCore(taskPerCore);
    tiling.set_taskCompNum(taskCompNum);
    tiling.set_taskCoreTail(taskCoreTail);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForGeometricKernelAttnGrad(gert::InferShapeContext* context)
{
    const gert::Shape* valueShape = context->GetInputShape(POS_INPUT_VALUE);
    const gert::Shape* attnWeightsShape = context->GetInputShape(POS_INPUT_ATTN_WEIGHTS);
    if (valueShape == nullptr || attnWeightsShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* gradValueShape = context->GetOutputShape(POS_OUTPUT_GRAD_VALUE);
    gert::Shape* gradAttnWeightsShape = context->GetOutputShape(POS_OUTPUT_GRAD_ATTN_WEIGHTS);
    if ((gradValueShape == nullptr) || (gradAttnWeightsShape == nullptr)) {
        return ge::GRAPH_FAILED;
    }

    *gradValueShape = *valueShape;
    *gradAttnWeightsShape = *attnWeightsShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForGeometricKernelAttnGrad(gert::InferDataTypeContext* context)
{
    const ge::DataType value_dtype = context->GetInputDataType(POS_INPUT_VALUE);
    const ge::DataType attn_weights_dtype = context->GetInputDataType(POS_INPUT_ATTN_WEIGHTS);
    context->SetOutputDataType(POS_OUTPUT_GRAD_VALUE, value_dtype);
    context->SetOutputDataType(POS_OUTPUT_GRAD_ATTN_WEIGHTS, attn_weights_dtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class GeometricKernelAttnGrad : public OpDef {
public:
    explicit GeometricKernelAttnGrad(const char* name) : OpDef(name)
    {
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("spatial_shapes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("level_start_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("sampling_locations")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("attn_weights")
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
        this->Output("grad_value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_attn_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForGeometricKernelAttnGrad)
            .SetInferDataType(ge::InferDataTypeForGeometricKernelAttnGrad);
        this->AICore().SetTiling(optiling::TilingFuncForGeometricKernelAttnGrad);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GeometricKernelAttnGrad);
} // namespace ops
