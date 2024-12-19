/*
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
*/
#include "gaussian_tiling.h"
#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace {
const uint32_t NUM_OBJS_IDX = 1;
const uint32_t FACTOR_IDX = 0;
const uint32_t OVERLAP_IDX = 1;
const uint32_t MIN_RADIUS_IDX = 2;
const uint32_t VOXEL_SIZEX_IDX = 3;
const uint32_t VOXEL_SIZEY_IDX = 4;
const uint32_t PC_RANGEX_IDX = 5;
const uint32_t PC_RANGEY_IDX = 6;
const uint32_t MAP_SIZEX_IDX = 7;
const uint32_t MAP_SIZEY_IDX = 8;
const uint32_t NORM_BBOX_IDX = 9;

const uint32_t OUTPUT_CENTER_INT = 0;
const uint32_t OUTPUT_RADIUS = 1;
const uint32_t OUTPUT_MASK = 2;
const uint32_t OUTPUT_IND = 3;
const uint32_t OUTPUT_SUBXY = 4;
const uint32_t OUTPUT_BOX_DIM = 5;
const uint32_t OUTPUT_SIN_ROT = 6;
const uint32_t OUTPUT_COS_ROT = 7;

constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t SIZE_OF_FP32 = 4;
constexpr uint32_t ALIGN_NUM = BYTE_BLOCK / SIZE_OF_FP32;
constexpr int32_t MAX_OBJS = 500;
} // namespace

namespace optiling {
static ge::graphStatus TilingFuncForGaussian(gert::TilingContext* context)
{
    GaussianTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto boxesShape = context->GetInputShape(0)->GetStorageShape();
    if (context->GetInputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int32_t numObjs = boxesShape.GetDim(NUM_OBJS_IDX);
    numObjs = std::min(numObjs, MAX_OBJS);
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto outSizeFactorPtr = attrs->GetAttrPointer<int32_t>(FACTOR_IDX);
    auto gaussianOverlapPtr = attrs->GetAttrPointer<float>(OVERLAP_IDX);
    auto minRadiusPtr = attrs->GetAttrPointer<int32_t>(MIN_RADIUS_IDX);
    auto voxelSizeXPtr = attrs->GetAttrPointer<float>(VOXEL_SIZEX_IDX);
    auto voxelSizeYPtr = attrs->GetAttrPointer<float>(VOXEL_SIZEY_IDX);
    auto pcRangeXPtr = attrs->GetAttrPointer<float>(PC_RANGEX_IDX);
    auto pcRangeYPtr = attrs->GetAttrPointer<float>(PC_RANGEY_IDX);
    auto mapSizeXPtr = attrs->GetAttrPointer<int32_t>(MAP_SIZEX_IDX);
    auto mapSizeYPtr = attrs->GetAttrPointer<int32_t>(MAP_SIZEY_IDX);
    auto normBboxPtr = attrs->GetAttrPointer<bool>(NORM_BBOX_IDX);
    if (outSizeFactorPtr == nullptr || gaussianOverlapPtr == nullptr || minRadiusPtr == nullptr ||
        voxelSizeXPtr == nullptr || voxelSizeYPtr == nullptr || pcRangeXPtr == nullptr || pcRangeYPtr == nullptr ||
        mapSizeXPtr == nullptr || mapSizeYPtr == nullptr || normBboxPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int32_t outSizeFactor = *outSizeFactorPtr;
    float gaussianOverlap = *gaussianOverlapPtr;
    int32_t minRadius = *minRadiusPtr;
    float voxelSizeX = *voxelSizeXPtr;
    float voxelSizeY = *voxelSizeYPtr;
    float pcRangeX = *pcRangeXPtr;
    float pcRangeY = *pcRangeYPtr;
    int32_t featureMapSizeX = *mapSizeXPtr;
    int32_t featureMapSizeY = *mapSizeYPtr;
    bool normBbox = *normBboxPtr;

    coreNum = std::min(static_cast<int32_t>(coreNum), numObjs);
    uint32_t average = numObjs / coreNum;
    uint32_t formerNum = numObjs % coreNum;
    uint32_t coreData = AlignUp(average + 1, ALIGN_NUM);
    context->SetBlockDim(coreNum);

    tiling.set_out_size_factor(outSizeFactor);
    tiling.set_gaussian_overlap(gaussianOverlap);
    tiling.set_min_radius(minRadius);
    tiling.set_voxel_size_x(voxelSizeX);
    tiling.set_voxel_size_y(voxelSizeY);
    tiling.set_pc_range_x(pcRangeX);
    tiling.set_pc_range_y(pcRangeY);
    tiling.set_feature_map_size_x(featureMapSizeX);
    tiling.set_feature_map_size_y(featureMapSizeY);
    tiling.set_num_objs(numObjs);
    tiling.set_norm_bbox(normBbox);
    tiling.set_core_data(coreData);
    tiling.set_average(average);
    tiling.set_former_num(formerNum);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForGaussian(gert::InferShapeContext* context)
{
    const gert::Shape* boxesShape = context->GetInputShape(0);
    if (boxesShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int32_t numObjs = boxesShape->GetDim(NUM_OBJS_IDX);
    numObjs = std::min(numObjs, MAX_OBJS);
    gert::Shape* centerIntShape = context->GetOutputShape(OUTPUT_CENTER_INT);
    gert::Shape* radiusShape = context->GetOutputShape(OUTPUT_RADIUS);
    gert::Shape* maskShape = context->GetOutputShape(OUTPUT_MASK);
    gert::Shape* indShape = context->GetOutputShape(OUTPUT_IND);
    gert::Shape* subXYShape = context->GetOutputShape(OUTPUT_SUBXY);
    gert::Shape* logBoxDimShape = context->GetOutputShape(OUTPUT_BOX_DIM);
    gert::Shape* sinRotShape = context->GetOutputShape(OUTPUT_SIN_ROT);
    gert::Shape* cosRotShape = context->GetOutputShape(OUTPUT_COS_ROT);
    if (centerIntShape == nullptr || radiusShape == nullptr || maskShape == nullptr || indShape == nullptr ||
        subXYShape == nullptr || logBoxDimShape == nullptr || sinRotShape == nullptr || cosRotShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    *centerIntShape = {2, numObjs};
    *subXYShape = {2, MAX_OBJS};
    *logBoxDimShape = {3, numObjs};
    radiusShape->AppendDim(numObjs);
    maskShape->AppendDim(MAX_OBJS);
    indShape->AppendDim(MAX_OBJS);
    sinRotShape->AppendDim(MAX_OBJS);
    cosRotShape->AppendDim(MAX_OBJS);

    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeForGaussian(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(4, ge::DT_FLOAT);
    context->SetOutputDataType(5, ge::DT_FLOAT);
    context->SetOutputDataType(6, ge::DT_FLOAT);
    context->SetOutputDataType(7, ge::DT_FLOAT);
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_INT32);
    context->SetOutputDataType(2, ge::DT_UINT8);
    context->SetOutputDataType(3, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Gaussian : public OpDef {
public:
    explicit Gaussian(const char* name) : OpDef(name)
    {
        this->Input("boxes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("center_int")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("radius")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("ind")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("sub_xy")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("log_box_dim")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("sin_rot")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("cos_rot")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("out_size_factor").AttrType(REQUIRED).Int();
        this->Attr("gaussian_overlap").AttrType(REQUIRED).Float();
        this->Attr("min_radius").AttrType(REQUIRED).Int();
        this->Attr("voxel_size_x").AttrType(REQUIRED).Float();
        this->Attr("voxel_size_y").AttrType(REQUIRED).Float();
        this->Attr("pc_range_x").AttrType(REQUIRED).Float();
        this->Attr("pc_range_y").AttrType(REQUIRED).Float();
        this->Attr("feature_map_size_x").AttrType(REQUIRED).Int();
        this->Attr("feature_map_size_y").AttrType(REQUIRED).Int();
        this->Attr("norm_bbox").AttrType(REQUIRED).Bool();
        this->Attr("with_velocity").AttrType(REQUIRED).Bool();

        this->SetInferShape(ge::InferShapeForGaussian).SetInferDataType(ge::InferDataTypeForGaussian);

        this->AICore().SetTiling(optiling::TilingFuncForGaussian);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(Gaussian);
} // namespace ops