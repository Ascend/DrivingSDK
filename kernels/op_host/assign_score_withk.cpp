/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#include "assign_score_withk_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

constexpr size_t BATCH_IDX = 0;
constexpr size_t NSOURCE_IDX = 1;
constexpr size_t NPOINT_IDX = 2;
constexpr size_t NWEIGHTS_IDX = 3;
constexpr size_t NNEIGHBORS_IDX = 4;
constexpr size_t NFEATURES_IDX = 5;
constexpr size_t AGG_IDX = 6;

namespace optiling {

/****************class impl*****************/
static ge::graphStatus AssignScoreWithkTilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape *pointShape = context->GetInputShape(0);
    const gert::StorageShape *centerShape = context->GetInputShape(1);
    const gert::StorageShape *scoreShape = context->GetInputShape(2);
    const gert::StorageShape *knnIdxShape = context->GetInputShape(3);
    const gert::RuntimeAttrs *attr = context->GetAttrs();
    auto platformInfoPtr = context->GetPlatformInfo();
    auto platformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    if ((pointShape == nullptr) || (centerShape == nullptr) || (scoreShape == nullptr) ||
        (knnIdxShape == nullptr) || (attr == nullptr) || (platformInfoPtr == nullptr) ||
        (context->GetInputDesc(0) == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    
    auto batchSizePtr = attr->GetAttrPointer<uint32_t>(BATCH_IDX);
    auto nsourcePtr = attr->GetAttrPointer<uint32_t>(NSOURCE_IDX);
    auto npointPtr = attr->GetAttrPointer<uint32_t>(NPOINT_IDX);
    auto numWeightsPtr = attr->GetAttrPointer<uint32_t>(NWEIGHTS_IDX);
    auto numNeighborsPtr = attr->GetAttrPointer<uint32_t>(NNEIGHBORS_IDX);
    auto numFeaturesPtr = attr->GetAttrPointer<uint32_t>(NFEATURES_IDX);
    auto aggregatePtr = attr->GetAttrPointer<uint32_t>(AGG_IDX);
    if ((!aggregatePtr) || (!batchSizePtr) || (!nsourcePtr) || (!npointPtr) || (!numWeightsPtr) ||
        (!numNeighborsPtr) || (!numFeaturesPtr)) {
        return ge::GRAPH_FAILED;
    }
    uint32_t batchSize = *batchSizePtr;
    uint32_t nsource = *nsourcePtr;
    uint32_t npoint = *npointPtr;
    uint32_t numWeights = *numWeightsPtr;
    uint32_t numNeighbors = *numNeighborsPtr;
    uint32_t numFeatures = *numFeaturesPtr;
    uint32_t aggregate = *aggregatePtr;
    uint32_t numCore = platformInfo.GetCoreNumAiv();
    if (numCore == 0) {
        return ge::GRAPH_FAILED;
    }

    size_t sysWorkspaceSize = platformInfo.GetLibApiWorkSpaceSize();
    size_t *currentWorkSpace = context->GetWorkspaceSizes(1);
    if (currentWorkSpace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkSpace[0] = sysWorkspaceSize;

    uint64_t npointPerCore = (static_cast<uint64_t>(batchSize) * numFeatures * npoint) / numCore;
    uint64_t npointRemained = (static_cast<uint64_t>(batchSize) * numFeatures * npoint) % numCore;

    AssignScoreWithkTilingData TilingData;
    TilingData.set_npointPerCore(npointPerCore);
    TilingData.set_npointRemained(npointRemained);
    TilingData.set_aggregate(aggregate);
    TilingData.set_batchSize(batchSize);
    TilingData.set_nsource(nsource);
    TilingData.set_npoint(npoint);
    TilingData.set_numWeights(numWeights);
    TilingData.set_numNeighbors(numNeighbors);
    TilingData.set_numFeatures(numFeatures);
    TilingData.set_numCore(numCore);
    context->SetBlockDim(numCore);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    TilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus AssignScoreWithkInferShape(gert::InferShapeContext *context)
{
    const gert::RuntimeAttrs *attr = context->GetAttrs();
    gert::Shape *outputShape = context->GetOutputShape(0);
    if ((attr == nullptr) || (outputShape == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    auto batchSizePtr = attr->GetAttrPointer<uint32_t>(BATCH_IDX);
    auto npointPtr = attr->GetAttrPointer<uint32_t>(NPOINT_IDX);
    auto numNeighborsPtr = attr->GetAttrPointer<uint32_t>(NNEIGHBORS_IDX);
    auto numFeaturesPtr = attr->GetAttrPointer<uint32_t>(NFEATURES_IDX);
    if ((!batchSizePtr) || (!npointPtr) || (!numNeighborsPtr) || (!numFeaturesPtr)) {
        return ge::GRAPH_FAILED;
    }
    uint32_t batchSize = *batchSizePtr;
    uint32_t npoint = *npointPtr;
    uint32_t numNeighbors = *numNeighborsPtr;
    uint32_t numFeatures = *numFeaturesPtr;

    outputShape->SetDimNum(4);
    *outputShape = {batchSize, numFeatures, npoint, numNeighbors};

    return GRAPH_SUCCESS;
}

static ge::graphStatus AssignScoreWithkInferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class AssignScoreWithk : public OpDef {
public:
    explicit AssignScoreWithk(const char* name) : OpDef(name)
    {
        this->Input("points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("centers")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("scores")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("knn_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("batch_size")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("nsource")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("npoint")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("num_weights")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("num_neighbors")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("num_features")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("aggregate")
            .AttrType(REQUIRED)
            .Int();

        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::AssignScoreWithkInferShape)
            .SetInferDataType(ge::AssignScoreWithkInferDataType);
        this->AICore().SetTiling(optiling::AssignScoreWithkTilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(AssignScoreWithk);
}