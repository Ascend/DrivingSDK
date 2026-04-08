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
const uint32_t B32_DATA_NUM_PER_REPEAT = 64;
const uint32_t B32_BYTE_SIZE = 4;
const uint64_t FULL_MASK = 0xffffffffffffffff;
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
    uint32_t aicNum = platformInfo.GetCoreNumAic();
    uint64_t numLevels = spatialShape.GetDim(NUM_LEVEL_DIM);
    uint64_t numPoints = attnWeightShape.GetDim(NUM_POINTS_DIM);
    uint64_t numHeads = attnWeightShape.GetDim(NUM_HEADS_DIM);
    uint64_t embedDims = valueShape.GetDim(EMBED_DIMS_DIM);
    uint64_t realLevels = attnWeightShape.GetDim(REAL_LEVEL_DIM);
    uint64_t batchSize = valueShape.GetDim(BATCH_SIZE_DIM);
    uint64_t numKeys = valueShape.GetDim(NUM_KEYS_DIM);
    uint64_t numQueries = attnWeightShape.GetDim(NUM_QUERIES_DIM);
    uint64_t embedDimsAlign = (embedDims + B32_DATA_NUM_PER_BLOCK - 1) / B32_DATA_NUM_PER_BLOCK * B32_DATA_NUM_PER_BLOCK;
    bool aligned = embedDims % B32_DATA_NUM_PER_BLOCK == 0;
    bool fastMode = (numHeads * numLevels * numPoints <= B32_DATA_NUM_PER_REPEAT) && (numHeads * numLevels * numPoints * embedDimsAlign <= 2048);

    context->SetTilingKey((aligned ? 1 : 0) * 10 + (fastMode ? 1 : 0));
    context->SetBlockDim(aicNum);

    constexpr size_t WORKSPACE_RSV_BYTE = 16 * 1024 * 1024;
    constexpr size_t GM_ALIGN = 512;
    constexpr size_t DB_NUM = 2;
    constexpr size_t VALID_FLAG_NUM = 8;
    constexpr size_t LOCATION_COORD_NUM = 4;
    uint64_t validFlagMaskLen = 256;
    uint32_t maxAlignedOneTaskNum = 8 * 256;
    uint32_t maxCornerCount = 32 * 256;
    size_t * workspaces = context->GetWorkspaceSizes(1);
    size_t workspaceOffset = WORKSPACE_RSV_BYTE;
    tiling.set_assembleWorkSpaceOffset(workspaceOffset);
    // cube组装之后的value 
    workspaceOffset = (workspaceOffset + coreNum * LOCATION_COORD_NUM * maxAlignedOneTaskNum * embedDimsAlign * B32_BYTE_SIZE + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    tiling.set_locationWorkSpaceOffset(workspaceOffset);
    // location 
    workspaceOffset = (workspaceOffset + DB_NUM * LOCATION_COORD_NUM * maxAlignedOneTaskNum * sizeof(float) * coreNum + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    tiling.set_validFlagWorkSpaceOffset(workspaceOffset);
    //  validFlag
    workspaceOffset = (workspaceOffset + DB_NUM * validFlagMaskLen * VALID_FLAG_NUM * coreNum + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    tiling.set_zeroWorkSpaceOffset(workspaceOffset);
    // 全0矩阵
    workspaceOffset =
        (workspaceOffset + maxCornerCount * B32_BYTE_SIZE + GM_ALIGN) / GM_ALIGN * GM_ALIGN; 

    workspaceOffset += WORKSPACE_RSV_BYTE;
    workspaces[0] = (workspaceOffset - 0);

    tiling.set_batchSize(batchSize);
    tiling.set_numKeys(numKeys);
    tiling.set_numHeads(numHeads);
    tiling.set_embedDims(embedDims);
    tiling.set_numLevels(numLevels);
    tiling.set_numQueries(numQueries);
    tiling.set_numPoints(numPoints);
    tiling.set_coreNum(coreNum);
    tiling.set_aicNum(aicNum);
    tiling.set_realLevels(realLevels);

    MX_DRIVING_LOGI(
        "MultiScaleDeformableAttn's tiling: batchSize=%d, numKeys=%d, numHeads=%d, embedDims=%d, numLevels=%d,numQueries=%d, numPoints=%d, coreNum=%d, pointLoops=%d,realLevels=%d",
        tiling.get_batchSize(), tiling.get_numKeys(), tiling.get_numHeads(), tiling.get_embedDims(),
        tiling.get_numLevels(), tiling.get_numQueries(), tiling.get_numPoints(), tiling.get_coreNum(),
        tiling.get_realLevels());

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

    yShape->AppendDim(valueShape->GetDim(BATCH_SIZE_DIM));
    yShape->AppendDim(samplingLocationsShape->GetDim(NUM_QUERIES_DIM));
    yShape->AppendDim(valueShape->GetDim(NUM_HEADS_DIM) * valueShape->GetDim(EMBED_DIMS_DIM));

    return GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForMultiScaleDeformableAttn(gert::InferDataTypeContext* context)
{
    CHECK_NULLPTR(context);
    const ge::DataType valueDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, valueDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MultiScaleDeformableAttn).InferShape(InferShapeForMultiScaleDeformableAttn).InferDataType(InferDataTypeForMultiScaleDeformableAttn);
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
