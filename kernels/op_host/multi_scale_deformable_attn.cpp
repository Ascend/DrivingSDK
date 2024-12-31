/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
 */
#include <cstdint>

#include "ge/utils.h"
#include "log/log.h"
#include "multi_scale_deformable_attn_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace {
const uint32_t INPUT_VALUE = 0;
const uint32_t INPUT_SPATIAL_SHAPE = 1;
const uint32_t INPUT_ATTN_WEIGHT = 4;
const uint32_t BATCH_SIZE_DIM = 0;
const uint32_t NUM_KEYS_DIM = 1;
const uint32_t NUM_HEADS_DIM = 2;
const uint32_t EMBED_DIMS_DIM = 3;
const uint32_t NUM_LEVEL_DIM = 0;
const uint32_t REAL_LEVEL_DIM = 3;
const uint32_t NUM_QUERIES_DIM = 1;
const uint32_t NUM_POINTS_DIM = 4;
const uint32_t B32_DATA_NUM_PER_BLOCK = 8;

std::tuple<uint32_t, uint32_t> GroupPoints(uint32_t numPoints)
{
    for (uint32_t p = 8; p >= 2; p--) {
        if (numPoints % p == 0) {
            return std::make_tuple(p, numPoints / p);
        }
    }
    return std::make_tuple(1, numPoints);
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFuncForMultiScaleDeformableAttn(gert::TilingContext* context)
{
    CHECK_NULLPTR(context);
    MultiScaleDeformableAttnTilingData tiling;

    auto valueTensorPtr = context->GetInputTensor(INPUT_VALUE);
    auto spatialTensorPtr = context->GetInputTensor(INPUT_SPATIAL_SHAPE);
    auto attnWeightTensorPtr = context->GetInputTensor(INPUT_ATTN_WEIGHT);
    CHECK_NULLPTR(valueTensorPtr);
    CHECK_NULLPTR(spatialTensorPtr);
    CHECK_NULLPTR(attnWeightTensorPtr);
    auto valueShape = valueTensorPtr->GetStorageShape();
    auto spatialShape = spatialTensorPtr->GetStorageShape();
    auto attnWeightShape = attnWeightTensorPtr->GetStorageShape();
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platformInfo.GetCoreNumAiv();
    context->SetBlockDim(coreNum);

    uint64_t numLevels = spatialShape.GetDim(NUM_LEVEL_DIM);
    uint64_t numPoints = attnWeightShape.GetDim(NUM_POINTS_DIM);
    uint64_t numHeads = attnWeightShape.GetDim(NUM_HEADS_DIM);
    uint64_t embedDims = valueShape.GetDim(EMBED_DIMS_DIM);
    auto groups = GroupPoints(numPoints);
    bool aligned = embedDims % B32_DATA_NUM_PER_BLOCK == 0;
    uint32_t point = std::get<0>(groups);
    uint32_t pointLoops = std::get<1>(groups);

    context->SetTilingKey(aligned ? point * 10 + 1 : point * 10);

    tiling.set_batchSize(valueShape.GetDim(BATCH_SIZE_DIM));
    tiling.set_numKeys(valueShape.GetDim(NUM_KEYS_DIM));
    tiling.set_numHeads(numHeads);
    tiling.set_embedDims(embedDims);
    tiling.set_numLevels(numLevels);
    tiling.set_numQueries(attnWeightShape.GetDim(NUM_QUERIES_DIM));
    tiling.set_numPoints(numPoints);
    tiling.set_coreNum(coreNum);
    tiling.set_pointLoops(pointLoops);
    tiling.set_realLevels(attnWeightShape.GetDim(REAL_LEVEL_DIM));
    MX_DRIVING_LOGI(
        "MultiScaleDeformableAttn's tiling: batchSize=%d, numKeys=%d, numHeads=%d, embedDims=%d, numLevels=%d,numQueries=%d, numPoints=%d, coreNum=%d, pointLoops=%d,realLevels=%d",
        tiling.get_batchSize(), tiling.get_numKeys(), tiling.get_numHeads(), tiling.get_embedDims(),
        tiling.get_numLevels(), tiling.get_numQueries(), tiling.get_numPoints(), tiling.get_coreNum(),
        tiling.get_pointLoops(), tiling.get_realLevels());

    ADD_TILING_DATA(context, tiling)

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShapeForMultiScaleDeformableAttn(gert::InferShapeContext* context)
{
    CHECK_NULLPTR(context);
    const gert::Shape* valueShape = context->GetInputShape(0);
    const gert::Shape* samplingLocationsShape = context->GetInputShape(3);
    gert::Shape* yShape = context->GetOutputShape(0);
    CHECK_NULLPTR(valueShape)
    CHECK_NULLPTR(samplingLocationsShape)
    CHECK_NULLPTR(yShape)

    yShape->SetDimNum(3);
    yShape->AppendDim(valueShape->GetDim(0));
    yShape->AppendDim(samplingLocationsShape->GetDim(1));
    yShape->AppendDim(valueShape->GetDim(1) * valueShape->GetDim(3));

    return GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForMultiScaleDeformableAttn(gert::InferDataTypeContext* context)
{
    CHECK_NULLPTR(context);
    const ge::DataType valueDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, valueDtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class MultiScaleDeformableAttn : public OpDef {
public:
    explicit MultiScaleDeformableAttn(const char* name) : OpDef(name)
    {
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value_spatial_shapes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value_level_start_index")
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
        this->Input("attention_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForMultiScaleDeformableAttn)
            .SetInferDataType(ge::InferDataTypeForMultiScaleDeformableAttn);

        this->AICore().SetTiling(optiling::TilingFuncForMultiScaleDeformableAttn);

        OpAICoreConfig aiConfig;
        aiConfig.ExtendCfgInfo("enableVectorCore.flag", "false");
        aiConfig.DynamicCompileStaticFlag(true);
        this->AICore().AddConfig("ascend310p", aiConfig);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(MultiScaleDeformableAttn);
} // namespace ops
