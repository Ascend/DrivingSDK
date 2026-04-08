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
    uint32_t nPoint;
    uint32_t nSource;
    bool isFromKnn;
    uint32_t coreNum;
    int32_t k;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape *xyzShape = context->GetInputShape(0);
    const gert::StorageShape *centerXyzShape = context->GetInputShape(1);
    const gert::RuntimeAttrs *attr = context->GetAttrs();
    auto platformInfoPtr = context->GetPlatformInfo();
    if ((xyzShape == nullptr) || (centerXyzShape == nullptr) || (attr == nullptr) || (platformInfoPtr == nullptr) ||
        (context->GetInputDesc(0) == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    if (attr->GetAttrPointer<uint32_t>(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    batch = centerXyzShape->GetStorageShape().GetDim(0);
    nPoint = centerXyzShape->GetStorageShape().GetDim(1);
    nSource = xyzShape->GetStorageShape().GetDim(2);
    isFromKnn = *attr->GetAttrPointer<bool>(0);
    k = *attr->GetAttrPointer<int32_t>(1);
    coreNum = platformInfo.GetCoreNumAiv();
    if (coreNum == 0) {
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
    TilingData.set_nPoint(nPoint);
    TilingData.set_nSource(nSource);
    TilingData.set_isFromKnn(isFromKnn);
    TilingData.set_coreNum(coreNum);
    TilingData.set_k(k);
    context->SetBlockDim(coreNum);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    TilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InfershapeForKnn(gert::InferShapeContext *context)
{
    const gert::Shape *xyzShape = context->GetInputShape(0);
    const gert::Shape *centerXyzShape = context->GetInputShape(1);
    gert::Shape *distShape = context->GetOutputShape(0);
    gert::Shape *idxShape = context->GetOutputShape(1);
    const gert::RuntimeAttrs *attr = context->GetAttrs();
    uint32_t batch;
    uint32_t nPoint;
    if ((xyzShape == nullptr) || (centerXyzShape == nullptr) || (distShape == nullptr) || (idxShape == nullptr) || (attr == nullptr)) {
            return ge::GRAPH_FAILED;
    }
    if ((xyzShape->GetDimNum() != 3) || ((centerXyzShape->GetDimNum() != 3))) { // 3 : input dim is 3
        return ge::GRAPH_FAILED;
    }
    batch = centerXyzShape->GetDim(0);
    nPoint = centerXyzShape->GetDim(1);
    const int32_t k = *attr->GetAttrPointer<int32_t>(1);

    distShape->SetDimNum(3);
    distShape->SetDim(0, batch);
    distShape->SetDim(1, nPoint);
    distShape->SetDim(2, k);
    
    idxShape->SetDimNum(3);
    idxShape->SetDim(0, batch);
    idxShape->SetDim(1, nPoint);
    idxShape->SetDim(2, k);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForKnn(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_INT32);
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
        this->Attr("k")
            .AttrType(REQUIRED)
            .Int();
        this->Output("dist")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InfershapeForKnn)
            .SetInferDataType(ge::InferDataTypeForKnn);
        this->AICore().SetTiling(optiling::TilingForKnn);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};

OP_ADD(Knn);
}