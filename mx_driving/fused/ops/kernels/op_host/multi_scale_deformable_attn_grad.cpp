/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
 */
#include "ge/utils.h"
#include "multi_scale_deformable_attn_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
using namespace ge;
using namespace std;

namespace {
const uint32_t INPUT_VALUE = 0;
const uint32_t INPUT_SPATIAL_SHAPE = 1;
const uint32_t INPUT_LOCATION = 3;
const uint32_t OUTPUT_GRAD_VALUE = 0;
const uint32_t OUTPUT_GRAD_LOCATION = 1;
const uint32_t OUTPUT_GRAD_WEIGHT = 2;
const uint32_t INPUT_ATTN_WEIGHT = 4;
const uint32_t BATCH_SIZE_DIM = 0;
const uint32_t NUM_KEYS_DIM = 1;
const uint32_t NUM_HEADS_DIM = 2;
const uint32_t EMBED_DIMS_DIM = 3;
const uint32_t NUM_LEVEL_DIM = 0;
const uint32_t REAL_LEVEL_DIM = 3;
const uint32_t NUM_QUERIES_DIM = 1;
const uint32_t NUM_POINTS_DIM = 4;
const uint32_t B32_DATA_NUM_PER_BLOCK = 4;

// the points can be grouped into 2, 4 or 8 points per block
// the numPoints has to be even, except 1
std::tuple<uint32_t, uint32_t> GroupPoints(uint32_t numPoints)
{
    if (numPoints % 8 == 0) {
        return std::make_tuple(8, numPoints / 8);
    }
    if (numPoints % 4 == 0) {
        return std::make_tuple(4, numPoints / 4);
    }
    if (numPoints % 2 == 0) {
        return std::make_tuple(2, numPoints / 2);
    }
    return std::make_tuple(1, numPoints);
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFuncForMultiScaleDeformableAttnGrad(gert::TilingContext* context)
{
    CHECK_NULLPTR(context);
    MultiScaleDeformableAttnGradTilingData tiling;

    auto valueTensorPtr = context->GetInputTensor(INPUT_VALUE);
    auto spatialTensorPtr = context->GetInputTensor(INPUT_SPATIAL_SHAPE);
    auto attnWeightTensorPtr = context->GetInputTensor(INPUT_ATTN_WEIGHT);
    CHECK_NULLPTR(valueTensorPtr);
    CHECK_NULLPTR(spatialTensorPtr);
    CHECK_NULLPTR(attnWeightTensorPtr);

    auto valueShape = valueTensorPtr->GetStorageShape();
    auto spatialShape = spatialTensorPtr->GetStorageShape();
    auto attnWeightShape = attnWeightTensorPtr->GetStorageShape();

    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendPlatformInfo.GetCoreNumAiv();
    context->SetBlockDim(coreNum);

    uint64_t batchSize = valueShape.GetDim(BATCH_SIZE_DIM);
    uint64_t numKeys = valueShape.GetDim(NUM_KEYS_DIM);
    uint64_t numHeads = attnWeightShape.GetDim(NUM_HEADS_DIM);
    uint64_t embedDims = valueShape.GetDim(EMBED_DIMS_DIM);
    uint64_t numQueries = attnWeightShape.GetDim(NUM_QUERIES_DIM);
    uint64_t numPoints = attnWeightShape.GetDim(NUM_POINTS_DIM);
    uint64_t numLevels = spatialShape.GetDim(NUM_LEVEL_DIM);
    bool optPoint = numLevels <= 8 && numHeads <= 8 && (embedDims == 16 || embedDims == 32 || embedDims == 64) &&
                        (numPoints % 2 == 0 || numPoints == 1);
    uint32_t pointLoops = 0;
    uint32_t point = 0;
    if (optPoint) {
        auto groups = GroupPoints(numPoints);
        pointLoops = std::get<1>(groups);
        point = std::get<0>(groups);
    }

    context->SetTilingKey(optPoint ? (embedDims / 16) * 1000 + point : 0);

    tiling.set_batchSize(batchSize);
    tiling.set_numKeys(numKeys);
    tiling.set_numHeads(numHeads);
    tiling.set_embedDims(embedDims);
    tiling.set_numLevels(numLevels);
    tiling.set_numQueries(numQueries);
    tiling.set_numPoints(numPoints);
    tiling.set_coreNum(coreNum);
    tiling.set_pointLoops(pointLoops);
    tiling.set_realLevels(attnWeightShape.GetDim(REAL_LEVEL_DIM));

    ADD_TILING_DATA(context, tiling);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    CHECK_NULLPTR(currentWorkspace);
    currentWorkspace[0] = 16 * 1024 * 1024;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForMultiScaleDeformableAttnGrad(gert::InferShapeContext* context)
{
    CHECK_NULLPTR(context);
    const gert::Shape* valueShape = context->GetInputShape(INPUT_VALUE);
    const gert::Shape* weightShape = context->GetInputShape(INPUT_ATTN_WEIGHT);
    const gert::Shape* spatialShape = context->GetInputShape(INPUT_SPATIAL_SHAPE);
    gert::Shape* gradValueShape = context->GetOutputShape(OUTPUT_GRAD_VALUE);
    gert::Shape* gradLocationShape = context->GetOutputShape(OUTPUT_GRAD_LOCATION);
    gert::Shape* gradWeightShape = context->GetOutputShape(OUTPUT_GRAD_WEIGHT);
    CHECK_NULLPTR(valueShape);
    CHECK_NULLPTR(weightShape);
    CHECK_NULLPTR(spatialShape);
    CHECK_NULLPTR(gradValueShape);
    CHECK_NULLPTR(gradLocationShape);
    CHECK_NULLPTR(gradWeightShape);

    int64_t batchSize = valueShape->GetDim(BATCH_SIZE_DIM);
    int64_t numKeys = valueShape->GetDim(NUM_KEYS_DIM);
    int64_t embedDims = valueShape->GetDim(EMBED_DIMS_DIM);

    int64_t numQueries = weightShape->GetDim(NUM_QUERIES_DIM);
    int64_t numLevels = spatialShape->GetDim(NUM_LEVEL_DIM);
    int64_t numHeads = weightShape->GetDim(NUM_HEADS_DIM);
    int64_t numPoints = weightShape->GetDim(NUM_POINTS_DIM);
    *gradValueShape = {batchSize, numKeys, numHeads, embedDims};
    *gradLocationShape = {batchSize, numQueries, numHeads, numLevels, 2, numPoints};
    *gradWeightShape = {batchSize, numQueries, numHeads, numLevels, numPoints};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMultiScaleDeformableAttnGrad(gert::InferDataTypeContext* context)
{
    CHECK_NULLPTR(context);
    const ge::DataType valueDtype = context->GetInputDataType(0);
    const ge::DataType samplingLocDtype = context->GetInputDataType(3);
    const ge::DataType attnWeightDtype = context->GetInputDataType(4);
    context->SetOutputDataType(0, valueDtype);
    context->SetOutputDataType(1, samplingLocDtype);
    context->SetOutputDataType(2, attnWeightDtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class MultiScaleDeformableAttnGrad : public OpDef {
public:
    explicit MultiScaleDeformableAttnGrad(const char* name) : OpDef(name)
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
        this->Input("sampling_loc")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("attn_weight")
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
        this->Output("grad_sampling_loc")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_attn_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForMultiScaleDeformableAttnGrad)
            .SetInferDataType(ge::InferDataTypeForMultiScaleDeformableAttnGrad);
        this->AICore().SetTiling(optiling::TilingFuncForMultiScaleDeformableAttnGrad);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(MultiScaleDeformableAttnGrad);
} // namespace ops
