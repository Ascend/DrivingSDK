/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#include "assign_score_withk_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "csrc/utils.h"

constexpr size_t BATCH_IDX = 0;
constexpr size_t NSOURCE_IDX = 1;
constexpr size_t NPOINT_IDX = 2;
constexpr size_t NWEIGHTS_IDX = 3;
constexpr size_t NNEIGHBORS_IDX = 4;
constexpr size_t NFEATURES_IDX = 5;
constexpr size_t AGG_IDX = 6;

constexpr size_t INPUT_GRADOUT_POSITION = 0;
constexpr size_t INPUT_POINTS_POSITION = 1;
constexpr size_t INPUT_CENTERS_POSITION = 2;
constexpr size_t INPUT_SCORES_POSITION = 3;
constexpr size_t INPUT_KNNIDX_POSITION = 4;

constexpr size_t OUTPUT_GRADSCORES_POSITION = 0;
constexpr size_t OUTPUT_GRADPOINTS_POSITION = 1;
constexpr size_t OUTPUT_GRADCENTERS_POSITION = 2;

namespace optiling {
/****************class impl*****************/
static ge::graphStatus AssignScoreWithkGradTilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape *gradOutShape = context->GetInputShape(INPUT_GRADOUT_POSITION);
    const gert::StorageShape *pointShape = context->GetInputShape(INPUT_POINTS_POSITION);
    const gert::StorageShape *centerShape = context->GetInputShape(INPUT_CENTERS_POSITION);
    const gert::StorageShape *scoreShape = context->GetInputShape(INPUT_SCORES_POSITION);
    const gert::StorageShape *knnIdxShape = context->GetInputShape(INPUT_KNNIDX_POSITION);
    const gert::RuntimeAttrs *attr = context->GetAttrs();
    auto platformInfoPtr = context->GetPlatformInfo();
    auto platformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    if ((gradOutShape == nullptr) || (pointShape == nullptr) || (centerShape == nullptr) ||
        (scoreShape == nullptr) || (knnIdxShape == nullptr) || (attr == nullptr) ||
        (platformInfoPtr == nullptr) || (context->GetInputDesc(0) == nullptr)) {
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

    uint64_t npointPerCore = (static_cast<uint64_t>(batchSize) * npoint) / numCore;
    uint64_t npointRemained = (static_cast<uint64_t>(batchSize) * npoint) % numCore;

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

    uint32_t dataAlign = 32 / sizeof(float);
    uint32_t featureAlign = AlignUp(numFeatures, dataAlign);
    std::vector<int64_t> shapeVector = {numWeights, featureAlign};
    ge::Shape srcShape(shapeVector);
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetUnPadMaxMinTmpSize(platformInfo, srcShape, sizeof(float), maxValue, minValue);
    const uint32_t localWorkSpaceSize = minValue;
    AscendC::UnPadTilingFunc(srcShape, localWorkSpaceSize, sizeof(float), TilingData.unpadTilingData);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    TilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus AssignScoreWithkGradInferShape(gert::InferShapeContext *context)
{
    const gert::RuntimeAttrs *attr = context->GetAttrs();
    gert::Shape *gradScoresShape = context->GetOutputShape(OUTPUT_GRADSCORES_POSITION);
    gert::Shape *gradPointsShape = context->GetOutputShape(OUTPUT_GRADPOINTS_POSITION);
    gert::Shape *gradCentersShape = context->GetOutputShape(OUTPUT_GRADCENTERS_POSITION);
    if ((attr == nullptr) || (gradScoresShape == nullptr) || (gradPointsShape == nullptr) || (gradCentersShape == nullptr)) {
        return ge::GRAPH_FAILED;
    }

    auto batchSizePtr = attr->GetAttrPointer<uint32_t>(BATCH_IDX);
    auto nsourcePtr = attr->GetAttrPointer<uint32_t>(NSOURCE_IDX);
    auto npointPtr = attr->GetAttrPointer<uint32_t>(NPOINT_IDX);
    auto numWeightsPtr = attr->GetAttrPointer<uint32_t>(NWEIGHTS_IDX);
    auto numNeighborsPtr = attr->GetAttrPointer<uint32_t>(NNEIGHBORS_IDX);
    auto numFeaturesPtr = attr->GetAttrPointer<uint32_t>(NFEATURES_IDX);
    if ((!batchSizePtr) || (!nsourcePtr) || (!npointPtr) || (!numWeightsPtr) || (!numNeighborsPtr) || (!numFeaturesPtr)) {
        return ge::GRAPH_FAILED;
    }
    uint32_t batchSize = *batchSizePtr;
    uint32_t nsource = *nsourcePtr;
    uint32_t npoint = *npointPtr;
    uint32_t numWeights = *numWeightsPtr;
    uint32_t numNeighbors = *numNeighborsPtr;
    uint32_t numFeatures = *numFeaturesPtr;

    gradScoresShape->SetDimNum(4);
    *gradScoresShape = {batchSize, npoint, numNeighbors, numWeights};
    gradPointsShape->SetDimNum(4);
    *gradPointsShape = {batchSize, nsource, numWeights, numFeatures};
    gradCentersShape->SetDimNum(4);
    *gradCentersShape = {batchSize, nsource, numWeights, numFeatures};

    return GRAPH_SUCCESS;
}

static ge::graphStatus AssignScoreWithkGradInferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class AssignScoreWithkGrad : public OpDef {
public:
    explicit AssignScoreWithkGrad(const char* name) : OpDef(name)
    {
        this->Input("grad_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
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

        this->Output("grad_scores")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_centers")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::AssignScoreWithkGradInferShape)
            .SetInferDataType(ge::AssignScoreWithkGradInferDataType);
        this->AICore().SetTiling(optiling::AssignScoreWithkGradTilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(AssignScoreWithkGrad);
}
