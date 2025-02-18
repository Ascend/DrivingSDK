/*
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
*/
#include "gaussian_tiling.h"
#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
using namespace std;
namespace {

} // namespace

namespace optiling {
constexpr uint32_t RESERVED_UB_SIZE = 8 * 1024;

static ge::graphStatus TilingFuncForGaussian(gert::TilingContext* context)
{
    CHECK_NULLPTR(context);
    auto platform = context->GetPlatformInfo();
    CHECK_NULLPTR(platform);
    auto platformInfo = platform_ascendc::PlatformAscendC(platform);
    uint32_t coreNum = platformInfo.GetCoreNumAiv();
    uint64_t ubSize;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    GaussianTilingData tiling;
    auto gtBoxesPtr = context->GetInputTensor(0);
    CHECK_NULLPTR(gtBoxesPtr);
    auto gtBoxesShape = gtBoxesPtr->GetStorageShape();
    uint32_t dimSize = gtBoxesShape.GetDim(0);
    uint32_t totalCoreTaskNum = gtBoxesShape.GetDim(1);
    uint32_t numObjs = totalCoreTaskNum;
    auto attrsPtr = context->GetAttrs();
    CHECK_NULLPTR(attrsPtr);
    uint32_t featureMapStride = *(attrsPtr->GetAttrPointer<int32_t>(0));
    float minOverLap = *(attrsPtr->GetAttrPointer<float>(1));
    uint32_t minRadius = *(attrsPtr->GetAttrPointer<int32_t>(2));
    uint32_t numMaxObjs = *(attrsPtr->GetAttrPointer<int32_t>(3));
    float voxelXSize = *(attrsPtr->GetAttrPointer<float>(4));
    float voxelYSize = *(attrsPtr->GetAttrPointer<float>(5));
    float prcX = *(attrsPtr->GetAttrPointer<float>(6));
    float prcY = *(attrsPtr->GetAttrPointer<float>(7));
    int32_t featureMapSizeX = *(attrsPtr->GetAttrPointer<int32_t>(8));
    int32_t featureMapSizeY = *(attrsPtr->GetAttrPointer<int32_t>(9));
    bool normBbox = *(attrsPtr->GetAttrPointer<bool>(10));
    bool flipAngle = *(attrsPtr->GetAttrPointer<bool>(11));
    if (totalCoreTaskNum > numMaxObjs) {
        totalCoreTaskNum = numMaxObjs;
    }

    uint32_t coreProcessTaskNum = AlignUp(Ceil(totalCoreTaskNum, coreNum), 8);
    uint32_t lastCoreProcessTaskNum = Tail(totalCoreTaskNum, coreProcessTaskNum);
    uint32_t usedCoreNum = Ceil(totalCoreTaskNum, coreProcessTaskNum);
    uint32_t singleProcessTaskNum = (ubSize - RESERVED_UB_SIZE) / 4 / (dimSize * 2 + 32);

    context->SetBlockDim(usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_numObjs(numObjs);
    tiling.set_totalCoreTaskNum(totalCoreTaskNum);
    tiling.set_coreProcessTaskNum(coreProcessTaskNum);
    tiling.set_lastCoreProcessTaskNum(lastCoreProcessTaskNum);
    tiling.set_singleProcessTaskNum(singleProcessTaskNum);
    // int32
    tiling.set_featureMapSizeX(featureMapSizeX);
    tiling.set_featureMapSizeY(featureMapSizeY);
    // float
    tiling.set_voxelXSize(voxelXSize);
    tiling.set_voxelYSize(voxelYSize);
    tiling.set_prcX(prcX);
    tiling.set_prcY(prcY);
    // int32
    tiling.set_featureMapStride(featureMapStride);
    tiling.set_numMaxObjs(numMaxObjs);
    tiling.set_minRadius(minRadius);
    // float
    tiling.set_minOverLap(minOverLap);
    // int32
    tiling.set_dimSize(dimSize);
    // bool
    tiling.set_normBbox(normBbox);
    tiling.set_flipAngle(flipAngle);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 16 * 1024 * 1024;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForGaussian(gert::InferShapeContext* context)
{
    CHECK_NULLPTR(context);
    const gert::Shape* gtBoxesShape = context->GetInputShape(0);
    auto attrsPtr = context->GetAttrs();
    CHECK_NULLPTR(gtBoxesShape);
    CHECK_NULLPTR(attrsPtr);
    gert::Shape* centerIntShape = context->GetOutputShape(0);
    gert::Shape* radiusShape = context->GetOutputShape(1);
    gert::Shape* maskShape = context->GetOutputShape(2);
    gert::Shape* indShape = context->GetOutputShape(3);
    gert::Shape* retBoxesShape = context->GetOutputShape(4);
    CHECK_NULLPTR(centerIntShape);
    CHECK_NULLPTR(radiusShape);
    CHECK_NULLPTR(maskShape);
    CHECK_NULLPTR(indShape);
    CHECK_NULLPTR(retBoxesShape);
    int64_t boxesMode = gtBoxesShape->GetDim(0);
    int64_t numObjs = gtBoxesShape->GetDim(1);
    uint32_t numMaxObjs = *(attrsPtr->GetAttrPointer<uint32_t>(3));
    if (numObjs > numMaxObjs) {
        numObjs = numMaxObjs;
    }
    *centerIntShape = {2, numObjs};
    *radiusShape = {numObjs};
    *maskShape = {numMaxObjs};
    *indShape = {numMaxObjs};
    *retBoxesShape = {numObjs, boxesMode + 1};
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeForGaussian(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_INT32);
    context->SetOutputDataType(2, ge::DT_UINT8);
    context->SetOutputDataType(3, ge::DT_INT32);
    context->SetOutputDataType(4, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Gaussian : public OpDef {
public:
    explicit Gaussian(const char* name) : OpDef(name)
    {
        this->Input("gt_boxes")
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
        this->Output("ret_boxes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("feature_map_stride").AttrType(REQUIRED).Int();
        this->Attr("gaussian_overlap").AttrType(REQUIRED).Float();
        this->Attr("min_radius").AttrType(REQUIRED).Int();
        this->Attr("num_max_objs").AttrType(REQUIRED).Int();
        this->Attr("voxel_size_x").AttrType(REQUIRED).Float();
        this->Attr("voxel_size_y").AttrType(REQUIRED).Float();
        this->Attr("pc_range_x").AttrType(REQUIRED).Float();
        this->Attr("pc_range_y").AttrType(REQUIRED).Float();
        this->Attr("feature_map_size_x").AttrType(REQUIRED).Int();
        this->Attr("feature_map_size_y").AttrType(REQUIRED).Int();
        this->Attr("norm_bbox").AttrType(REQUIRED).Bool();
        this->Attr("flip_angle").AttrType(REQUIRED).Bool();
        this->SetInferShape(ge::InferShapeForGaussian).SetInferDataType(ge::InferDataTypeForGaussian);

        this->AICore().SetTiling(optiling::TilingFuncForGaussian);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(Gaussian);
} // namespace ops