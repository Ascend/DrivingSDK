/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "roipoint_pool3d_forward_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

/*
 * points转置: (B, 3, N) 输入点
 * point_features转置: (B, C, N) 输入点特征
 * boxes3d: (B, M, 7) 边界框
 * pooled_features转置: (B, M, 3+C, num) 特征汇聚
 * pooled_empty_flag: (B, M) 空标志
*/
namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
const uint64_t TILING_KEY_FLOAT = 1;
const uint64_t TILING_KEY_HALF = 2;
const int32_t NUM_SAMPLED_POINTS = 512;
static ge::graphStatus TilingForRoipointPool3dForward(gert::TilingContext* context)
{
    RoipointPool3dForwardTilingData tiling;

    int32_t numSampledPoints = NUM_SAMPLED_POINTS;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        auto gap = attrs->GetAttrPointer<int32_t>(0);
        if (gap != nullptr) {
            numSampledPoints = *gap;
            if (numSampledPoints <= 0) {
                numSampledPoints = NUM_SAMPLED_POINTS;
            }
        }
    }
    tiling.set_numSampledPoints(static_cast<uint32_t>(numSampledPoints));

    auto pointsTensor = context->GetInputTensor(0);
    auto pointFeaturesTensor = context->GetInputTensor(1);
    auto boxes3DTensor = context->GetInputTensor(2);
    if (pointsTensor == nullptr || pointFeaturesTensor == nullptr || boxes3DTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t batchSize = pointsTensor->GetStorageShape().GetDim(0);
    uint32_t pointNum = pointsTensor->GetStorageShape().GetDim(2);
    uint32_t featureLen = pointFeaturesTensor->GetStorageShape().GetDim(1);
    uint32_t boxesNum = boxes3DTensor->GetStorageShape().GetDim(1);
    tiling.set_batchSize(batchSize);
    tiling.set_pointNum(pointNum);
    tiling.set_featureLen(featureLen);
    tiling.set_boxesNum(boxesNum);

    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    uint64_t ubSize = 0;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tiling.set_ubSize(ubSize);

    uint32_t format;
    auto tensorDesc = context->GetInputDesc(0);
    if (tensorDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dType = tensorDesc->GetDataType();
    if (dType == ge::DT_FLOAT) {
        format = 32 / sizeof(float);
        context->SetTilingKey(TILING_KEY_FLOAT);
    } else {
        format = 16;
        context->SetTilingKey(TILING_KEY_HALF);
    }
    uint32_t coreNum = ascendplatformInfo.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint16_t eachCoreBoxes = (batchSize * boxesNum - 1) / coreNum + 1;
    eachCoreBoxes = ((eachCoreBoxes + format - 1) / format) * format;
    coreNum = (batchSize * boxesNum - 1) / eachCoreBoxes + 1;
    tiling.set_eachCoreBoxes(eachCoreBoxes);
    context->SetBlockDim(coreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

// points转置: (B, 3, N) 输入点
// point_features转置: (B, C, N) 输入点特征
// boxes3d: (B, M, 7) 边界框
// pooled_features转置: (B, M, 3+C, num) 特征汇聚
// pooled_empty_flag: (B, M) 空标志
namespace ge {
const uint32_t POINTS_COORDINATE = 3;
const int32_t NUM_SAMPLED_POINTS = 512;
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* points_shape = context->GetInputShape(0);
    if (points_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* point_features_shape = context->GetInputShape(1);
    if (point_features_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* boxes3d_shape = context->GetInputShape(2);
    if (boxes3d_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* pooled_features_shape = context->GetOutputShape(0);
    if (pooled_features_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* pooled_empty_flag_shape = context->GetOutputShape(1);
    if (pooled_empty_flag_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int32_t numSampledPoints = NUM_SAMPLED_POINTS;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        numSampledPoints = *(attrs->GetAttrPointer<int32_t>(0));
    }

    pooled_features_shape->SetDimNum(0);
    pooled_features_shape->AppendDim(boxes3d_shape->GetDim(0));
    pooled_features_shape->AppendDim(boxes3d_shape->GetDim(1));
    pooled_features_shape->AppendDim(POINTS_COORDINATE + point_features_shape->GetDim(1));
    pooled_features_shape->AppendDim(numSampledPoints);
    pooled_empty_flag_shape->SetDimNum(0);
    pooled_empty_flag_shape->AppendDim(boxes3d_shape->GetDim(0));
    pooled_empty_flag_shape->AppendDim(boxes3d_shape->GetDim(1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForRoipointPool3dForward(gert::InferDataTypeContext* context)
{
    const ge::DataType value_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, value_dtype);
    context->SetOutputDataType(1, ge::DT_INT32);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class RoipointPool3dForward : public OpDef {
public:
    explicit RoipointPool3dForward(const char* name) : OpDef(name)
    {
        this->Input("points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("point_features")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("boxes3d")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("pooled_features")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("pooled_empty_flag")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("num_sampled_points")
            .AttrType(OPTIONAL).Int();

        this->SetInferShape(ge::InferShape)
             .SetInferDataType(ge::InferDataTypeForRoipointPool3dForward);

        this->AICore()
            .SetTiling(optiling::TilingForRoipointPool3dForward);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(RoipointPool3dForward);
}