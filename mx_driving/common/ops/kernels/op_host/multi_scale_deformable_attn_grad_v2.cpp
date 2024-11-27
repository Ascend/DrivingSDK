/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#include "multi_scale_deformable_attn_grad_tiling_v2.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace ge;
using namespace std;

namespace {
const uint32_t INPUT_VALUE = 0;
const uint32_t INPUT_SPATIAL_SHAPE = 1;
const uint32_t INPUT_LOCATION = 3;
const uint32_t OUTPUT_ATTN_WEIGHT = 2;
const uint32_t INPUT_ATTN_WEIGHT = 4;
const uint32_t BATCh_SIZE_DIM = 0;
const uint32_t NUM_KEYS_DIM = 2;
const uint32_t NUM_HEADS_DIM = 1;
const uint32_t EMBED_DIMS_DIM = 3;
const uint32_t NUM_LEVEL_DIM = 0;
const uint32_t NUM_QUERIES_DIM = 4;
const uint32_t NUM_POINTS_DIM = 3;
const uint32_t TILING_KEY_NP_TWO = 0;
const uint32_t TILING_KEY_NP_FOUR = 1;
const uint32_t TILING_KEY_NP_EIGHT = 2;
const uint32_t TILING_KEY_GENERIC = 3;
const uint64_t RESERVE_SAPCE = 1024;
const uint32_t DATA_ALIGN = 8;
} // namespace

namespace optiling {
static ge::graphStatus TilingFuncForMultiScaleDeformableAttnGradV2(gert::TilingContext* context)
{
    MultiScaleDeformableAttnGradTilingDataV2 tiling;

    auto valueTensorPtr = context->GetInputTensor(INPUT_VALUE);
    auto spatialTensorPtr = context->GetInputTensor(INPUT_SPATIAL_SHAPE);
    auto attnWeightTensorPtr = context->GetInputTensor(INPUT_ATTN_WEIGHT);
    if (valueTensorPtr == nullptr || spatialTensorPtr == nullptr || attnWeightTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto valueShape = valueTensorPtr->GetStorageShape();
    auto spatialShape = spatialTensorPtr->GetStorageShape();
    auto attnWeightShape = attnWeightTensorPtr->GetStorageShape();

    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    uint32_t coreNum = ascendplatformInfo.GetCoreNumAiv();
    context->SetBlockDim(coreNum);

    uint32_t batchSize = valueShape.GetDim(BATCh_SIZE_DIM);
    uint32_t numKeys = valueShape.GetDim(NUM_KEYS_DIM);
    uint32_t numHeads = attnWeightShape.GetDim(NUM_HEADS_DIM);
    uint32_t embedDims = valueShape.GetDim(EMBED_DIMS_DIM);
    uint32_t numLevels = spatialShape.GetDim(NUM_LEVEL_DIM);
    uint32_t numQueries = attnWeightShape.GetDim(NUM_QUERIES_DIM);
    uint32_t numPoints = attnWeightShape.GetDim(NUM_POINTS_DIM);

    uint64_t total_ub_size;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, total_ub_size);
    uint64_t ub_size = total_ub_size - RESERVE_SAPCE;
    uint32_t numLevelsAlign = (numLevels + DATA_ALIGN - 1) / DATA_ALIGN * DATA_ALIGN;
    uint32_t max_ub_num = (ub_size / 4 - 3 * numLevelsAlign - 8 * embedDims) / (15 + 13 * embedDims);
    max_ub_num = max_ub_num / DATA_ALIGN * DATA_ALIGN;
    tiling.set_batchSize(batchSize);
    tiling.set_numKeys(numKeys);
    tiling.set_numHeads(numHeads);
    tiling.set_embedDims(embedDims);
    tiling.set_numLevels(numLevels);
    tiling.set_numQueries(numQueries);
    tiling.set_numPoints(numPoints);
    tiling.set_maxUbNum(max_ub_num);
    tiling.set_coreNum(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 16 * 1024 * 1024;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForMultiScaleDeformableAttnGradV2(gert::InferShapeContext* context)
{
    const gert::Shape* value_shape = context->GetInputShape(0);
    if (value_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* sampling_locations_shape = context->GetInputShape(INPUT_LOCATION);
    if (sampling_locations_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* attn_weight_shape = context->GetInputShape(INPUT_ATTN_WEIGHT);
    if (attn_weight_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* grad_value_shape = context->GetOutputShape(0);
    gert::Shape* grad_sample_loc_shape = context->GetOutputShape(1);
    gert::Shape* grad_attn_weight_shape = context->GetOutputShape(OUTPUT_ATTN_WEIGHT);
    if ((grad_value_shape == nullptr) || (grad_sample_loc_shape == nullptr) || (grad_attn_weight_shape == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    *grad_value_shape = *value_shape;
    *grad_sample_loc_shape = *sampling_locations_shape;
    *grad_attn_weight_shape = *attn_weight_shape;
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class MultiScaleDeformableAttnGradV2 : public OpDef {
public:
    explicit MultiScaleDeformableAttnGradV2(const char* name) : OpDef(name)
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

        this->SetInferShape(ge::InferShapeForMultiScaleDeformableAttnGradV2);
        this->AICore().SetTiling(optiling::TilingFuncForMultiScaleDeformableAttnGradV2);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MultiScaleDeformableAttnGradV2);
} // namespace ops
