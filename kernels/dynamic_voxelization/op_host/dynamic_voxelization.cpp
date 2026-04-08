/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */

#include "dynamic_voxelization_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingForDynamicVox(gert::TilingContext* context)
{
    DynamicVoxTilingData tiling;
    // get core num
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t coreNum = ascendPlatformInfo.GetCoreNumAic();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    // get tiling param
    if (context->GetInputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ptsShape = context->GetInputShape(0)->GetStorageShape();
    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    float coorsMinX = *(attrsPtr->GetAttrPointer<float>(0));
    float coorsMinY = *(attrsPtr->GetAttrPointer<float>(1));
    float coorsMinZ = *(attrsPtr->GetAttrPointer<float>(2));
    float voxelX = *(attrsPtr->GetAttrPointer<float>(3));
    float voxelY = *(attrsPtr->GetAttrPointer<float>(4));
    float voxelZ = *(attrsPtr->GetAttrPointer<float>(5));
    int gridX = *(attrsPtr->GetAttrPointer<int32_t>(6));
    int gridY = *(attrsPtr->GetAttrPointer<int32_t>(7));
    int gridZ = *(attrsPtr->GetAttrPointer<int32_t>(8));

    // tiling by pts num
    uint32_t ptsNum = ptsShape.GetDim(1);
    uint32_t ptsFeature = ptsShape.GetDim(0);

    uint32_t ptsNumInCore = ptsNum / coreNum;
    uint32_t ptsNumInLastCore = ptsNum - ptsNumInCore * (coreNum - 1);

    // save param
    context->SetBlockDim(coreNum);
    tiling.set_core_num(coreNum);
    tiling.set_points_num_in_core(ptsNumInCore);
    tiling.set_points_num_in_last_core(ptsNumInLastCore);
    tiling.set_pts_num(ptsNum);
    tiling.set_pts_feature(ptsFeature);
    tiling.set_coors_min_x(coorsMinX);
    tiling.set_coors_min_y(coorsMinY);
    tiling.set_coors_min_z(coorsMinZ);
    tiling.set_grid_x(gridX);
    tiling.set_grid_y(gridY);
    tiling.set_grid_z(gridZ);
    tiling.set_voxel_x(voxelX);
    tiling.set_voxel_y(voxelY);
    tiling.set_voxel_z(voxelZ);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForDynamicVoxel(gert::InferShapeContext* context)
{
    if (context->GetInputShape(0) == nullptr || context->GetInputShape(1) ==nullptr || context->GetOutputShape(0)) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* ptsShape = context->GetInputShape(0);
    if (ptsShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* coorsShape = context->GetOutputShape(0);
    if (coorsShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    coorsShape->SetDimNum(0);
    coorsShape->AppendDim(3);
    coorsShape->AppendDim(ptsShape->GetDim(1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForDynamicVoxel(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    return GRAPH_SUCCESS;
}
} // namespace ge


namespace ops {
class DynamicVoxelization : public OpDef {
public:
    explicit DynamicVoxelization(const char* name) : OpDef(name)
    {
        this->Input("points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("coors")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("coors_min_x").Float();
        this->Attr("coors_min_y").Float();
        this->Attr("coors_min_z").Float();
        this->Attr("voxel_x").Float();
        this->Attr("voxel_y").Float();
        this->Attr("voxel_z").Float();
        this->Attr("grid_x").Int();
        this->Attr("grid_y").Int();
        this->Attr("grid_z").Int();

        this->SetInferShape(ge::InferShapeForDynamicVoxel)
            .SetInferDataType(ge::InferDataTypeForDynamicVoxel);

        this->AICore().SetTiling(optiling::TilingForDynamicVox);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(DynamicVoxelization);
} // namespace ops