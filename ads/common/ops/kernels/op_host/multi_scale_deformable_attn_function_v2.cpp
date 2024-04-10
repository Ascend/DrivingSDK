/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#include "multi_scale_deformable_attn_function_v2.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace std;
using namespace AscendC;

namespace optiling
{
    const uint32_t BLOCK_DIM = 8;
    const uint32_t TILE_NUM = 8;
    const uint32_t INPUT_VALUE = 0;
    const uint32_t INPUT_SPATIAL_SHAPE = 1;
    const uint32_t INPUT_ATTN_WEIGHT = 4;
    const uint32_t BATCh_SIZE_DIM = 0;
    const uint32_t NUM_KEYS_DIM = 2;
    const uint32_t NUM_HEADS_DIM = 2;
    const uint32_t EMBED_DIMS_DIM = 3;
    const uint32_t NUM_LEVEL_DIM = 0;
    const uint32_t NUM_QUERIES_DIM = 1;
    const uint32_t NUM_POINTS_DIM = 4;
    static ge::graphStatus TilingFuncForMultiScaleDeformableAttnFunctionV2(gert::TilingContext *context)
    {
        MultiScaleDeformableAttnFunctionV2TilingData tiling;

        auto valueTensorPtr = context->GetInputTensor(0);
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

        tiling.set_batchSize(valueShape.GetDim(BATCh_SIZE_DIM));
        tiling.set_numKeys(valueShape.GetDim(NUM_KEYS_DIM));
        tiling.set_numHeads(attnWeightShape.GetDim(NUM_HEADS_DIM));
        tiling.set_embedDims(valueShape.GetDim(EMBED_DIMS_DIM));
        tiling.set_numLevels(spatialShape.GetDim(NUM_LEVEL_DIM));
        tiling.set_numQueries(attnWeightShape.GetDim(NUM_QUERIES_DIM));
        tiling.set_numPoints(attnWeightShape.GetDim(NUM_POINTS_DIM));
        tiling.set_coreNum(coreNum);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge
{
    static ge::graphStatus InferShapeForMultiScaleDeformableAttnFunctionV2(gert::InferShapeContext *context)
    {
        const gert::Shape *valueShape = context->GetInputShape(0);
        if (valueShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape *samplingLocationsShape = context->GetInputShape(3);
        if (samplingLocationsShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        gert::Shape *y_shape = context->GetOutputShape(0);
        if (y_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        y_shape->SetDimNum(0);
        y_shape->AppendDim(valueShape->GetDim(0));
        y_shape->AppendDim(samplingLocationsShape->GetDim(1));
        y_shape->AppendDim(samplingLocationsShape->GetDim(2) * valueShape->GetDim(3));

        return GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataTypeForMultiScaleDeformableAttnFunctionV2(gert::InferDataTypeContext* context)
    {
        const ge::DataType value_dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, value_dtype);
        return GRAPH_SUCCESS;
    }
}

namespace ops
{
    class MultiScaleDeformableAttnFunctionV2 : public OpDef
    {
    public:
        explicit MultiScaleDeformableAttnFunctionV2(const char *name) : OpDef(name)
        {
            this->Input("value")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("value_spatial_shapes")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("value_level_start_index")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("sampling_locations")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("attention_weights")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("output")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->SetInferShape(ge::InferShapeForMultiScaleDeformableAttnFunctionV2)
                .SetInferDataType(ge::InferDataTypeForMultiScaleDeformableAttnFunctionV2);

            this->AICore()
                .SetTiling(optiling::TilingFuncForMultiScaleDeformableAttnFunctionV2);
            
            OpAICoreConfig aiConfig;
            aiConfig.ExtendCfgInfo("enableVectorCore.flag", "false");
            aiConfig.DynamicCompileStaticFlag(true);
            this->AICore().AddConfig("ascend310p", aiConfig);
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(MultiScaleDeformableAttnFunctionV2);
}
