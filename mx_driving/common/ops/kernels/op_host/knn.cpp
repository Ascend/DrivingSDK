/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */

#include "knn_tiling.h"
#include "common.h"

namespace optiling {
/****************class impl*****************/
static ge::graphStatus TilingForKnn(gert::TilingContext *context)
{
    uint32_t batch;
    uint32_t npoint;
    uint32_t nsource;
    bool is_from_knn;
    uint32_t core_num;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape *xyz_shape = context->GetInputShape(0);
    const gert::StorageShape *center_xyz_shape = context->GetInputShape(1);
    const gert::RuntimeAttrs *attr = context->GetAttrs();
    auto platformInfoPtr = context->GetPlatformInfo();
    ge::DataType dtype_;
    if ((xyz_shape == nullptr) || (center_xyz_shape == nullptr) || (attr == nullptr) || (platformInfoPtr == nullptr) ||
        (context->GetInputDesc(0) == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    if (attr->GetAttrPointer<uint32_t>(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    batch = center_xyz_shape->GetStorageShape().GetDim(0);
    npoint = center_xyz_shape->GetStorageShape().GetDim(1);
    nsource = xyz_shape->GetStorageShape().GetDim(2);
    is_from_knn = *attr->GetAttrPointer<bool>(0);
    core_num = platformInfo.GetCoreNumAiv();
    if (core_num == 0) {
        return ge::GRAPH_FAILED;
    }

    size_t sysWorkspaceSize = 16 * 1024 * 1024; // Alloc 16M workspace
    size_t *currentWorkSpace = context->GetWorkspaceSizes(1);
    if (currentWorkSpace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkSpace[0] = sysWorkspaceSize;

    KnnTilingData TilingData;
    TilingData.set_batch(batch);
    TilingData.set_npoint(npoint);
    TilingData.set_nsource(nsource);
    TilingData.set_is_from_knn(is_from_knn);
    TilingData.set_core_num(core_num);
    context->SetBlockDim(core_num);
    TilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InfershapeForKnn(gert::InferShapeContext *context)
{
    const gert::Shape *xyz_shape = context->GetInputShape(0);
    const gert::Shape *center_xyz_shape = context->GetInputShape(1);
    gert::Shape *dist_shape = context->GetOutputShape(0);
    const gert::RuntimeAttrs *attr = context->GetAttrs();
    uint32_t batch;
    uint32_t npoint;
    uint32_t nsource;
    if ((xyz_shape == nullptr) || (center_xyz_shape == nullptr) || (dist_shape == nullptr) || (attr == nullptr)) {
            return ge::GRAPH_FAILED;
    }
    const bool *is_from_knn = attr->GetAttrPointer<bool>(0);
    if (is_from_knn == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if ((xyz_shape->GetDimNum() != 3) || ((center_xyz_shape->GetDimNum() != 3))) { // 3 : input dim is 3
        return ge::GRAPH_FAILED;
    }
    batch = center_xyz_shape->GetDim(0);
    npoint = center_xyz_shape->GetDim(1);
    nsource = xyz_shape->GetDim(2);
    dist_shape->SetDimNum(3);
    dist_shape->SetDim(0, batch);
    dist_shape->SetDim(1, npoint);
    dist_shape->SetDim(2, nsource);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Knn : public OpDef {
public:
    explicit Knn(const char* name) : OpDef(name)
    {
        this->Input("xyz")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("center_xyz")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("is_from_knn")
            .AttrType(REQUIRED)
            .Bool();
        this->Output("dist")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InfershapeForKnn);
        this->AICore().SetTiling(optiling::TilingForKnn);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910c", aicore_config);
    }
};

OP_ADD(Knn);
}