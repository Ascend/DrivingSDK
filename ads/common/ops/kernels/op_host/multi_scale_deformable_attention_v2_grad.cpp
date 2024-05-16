/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
 */
#include "multi_scale_deformable_attention_v2_grad.h"

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
const uint32_t NUM_HEADS_DIM = 2;
const uint32_t EMBED_DIMS_DIM = 3;
const uint32_t NUM_LEVEL_DIM = 0;
const uint32_t NUM_QUERIES_DIM = 1;
const uint32_t NUM_POINTS_DIM = 4;
const uint32_t TILING_KEY_BEVFORMER = 0;
const uint32_t TILING_KEY_GENERIC = 1;
} // namespace

namespace optiling {
static ge::graphStatus TilingFuncForMultiScaleDeformableAttentionV2Grad(gert::TilingContext* context)
{
    MultiScaleDeformableAttentionV2GradTilingData tiling;

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

    tiling.set_batchSize(batchSize);
    tiling.set_numKeys(numKeys);
    tiling.set_numHeads(numHeads);
    tiling.set_embedDims(embedDims);
    tiling.set_numLevels(numLevels);
    tiling.set_numQueries(numQueries);
    tiling.set_numPoints(numPoints);
    
    if (numPoints == 8 && embedDims == 32) {
        context->SetTilingKey(TILING_KEY_BEVFORMER);
    } else {
        context->SetTilingKey(TILING_KEY_GENERIC);
    }

    tiling.set_coreNum(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 16 * 1024 * 1024;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForMultiScaleDeformableAttentionV2Grad(gert::InferShapeContext* context)
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
class MultiScaleDeformableAttentionV2Grad : public OpDef {
public:
    explicit MultiScaleDeformableAttentionV2Grad(const char* name) : OpDef(name)
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

        this->SetInferShape(ge::InferShapeForMultiScaleDeformableAttentionV2Grad);

        this->AICore().SetTiling(optiling::TilingFuncForMultiScaleDeformableAttentionV2Grad);

        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MultiScaleDeformableAttentionV2Grad);
} // namespace ops
