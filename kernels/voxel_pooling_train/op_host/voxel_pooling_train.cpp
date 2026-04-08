/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "voxel_pooling_train_tiling.h"


namespace {
constexpr uint32_t INDICES = 3;

constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t SIZE_OF_DATA = 4;
constexpr uint32_t BLOCK_INT32 = 8;

constexpr size_t B_IDX = 0;
constexpr size_t N_IDX = 1;
constexpr size_t C_IDX = 2;
constexpr size_t X_IDX = 3;
constexpr size_t Y_IDX = 4;
constexpr size_t Z_IDX = 5;

} // namespace

namespace optiling {
static ge::graphStatus TilingForVoxelPooling(gert::TilingContext* context)
{
    VoxelPoolingTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // get core num
    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t coreNum = ascendPlatformInfo.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    // get tiling param
    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int batchSize = *(attrsPtr->GetAttrPointer<int>(B_IDX));
    int numPoints = *(attrsPtr->GetAttrPointer<int>(N_IDX));
    int numChannels = *(attrsPtr->GetAttrPointer<int>(C_IDX));
    int numVoxelX = *(attrsPtr->GetAttrPointer<int>(X_IDX));
    int numVoxelY = *(attrsPtr->GetAttrPointer<int>(Y_IDX));
    int numVoxelZ = *(attrsPtr->GetAttrPointer<int>(Z_IDX));

    uint32_t alignNum = BYTE_BLOCK / SIZE_OF_DATA;
    uint32_t cAligned = (numChannels + alignNum - 1) / alignNum * alignNum;
    uint32_t indicesAligned = (INDICES + BLOCK_INT32 - 1) / BLOCK_INT32 * BLOCK_INT32;

    uint32_t average = batchSize * numPoints / coreNum;
    uint32_t taskLast = batchSize * numPoints % coreNum;
    uint32_t usedCoreNum = coreNum;

    if (average == 0) {
        usedCoreNum = taskLast;
    }
    // save param
    context->SetBlockDim(usedCoreNum);

    tiling.set_batchSize(batchSize);
    tiling.set_numPoints(numPoints);
    tiling.set_numChannels(numChannels);
    tiling.set_cAligned(cAligned);
    tiling.set_numVoxelX(numVoxelX);
    tiling.set_numVoxelY(numVoxelY);
    tiling.set_numVoxelZ(numVoxelZ);
    tiling.set_indicesAligned(indicesAligned);
    tiling.set_average(average);
    tiling.set_taskLast(taskLast);
    tiling.set_usedCoreNum(usedCoreNum);
    
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForVoxelPoolingTrain(gert::InferShapeContext* context)
{
    auto attrsPtr = context->GetAttrs();
    gert::Shape* outFeaturesShape = context->GetOutputShape(0);
    gert::Shape* posMemoShape = context->GetOutputShape(1);
    if (attrsPtr == nullptr || outFeaturesShape == nullptr || posMemoShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int batchSize = *(attrsPtr->GetAttrPointer<int>(B_IDX));
    int numPoints = *(attrsPtr->GetAttrPointer<int>(N_IDX));
    int numChannels = *(attrsPtr->GetAttrPointer<int>(C_IDX));
    int numVoxelX = *(attrsPtr->GetAttrPointer<int>(X_IDX));
    int numVoxelY = *(attrsPtr->GetAttrPointer<int>(Y_IDX));
    int numVoxelZ = *(attrsPtr->GetAttrPointer<int>(Z_IDX));

    outFeaturesShape->SetDimNum(0);
    outFeaturesShape->AppendDim(batchSize);
    outFeaturesShape->AppendDim(numVoxelY);
    outFeaturesShape->AppendDim(numVoxelX);
    outFeaturesShape->AppendDim(numChannels);

    posMemoShape->SetDimNum(0);
    posMemoShape->AppendDim(batchSize);
    posMemoShape->AppendDim(numPoints);
    posMemoShape->AppendDim(3);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForVoxelPoolingTrain(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_INT32);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class VoxelPoolingTrain : public OpDef {
public:
    explicit VoxelPoolingTrain(const char* name) : OpDef(name)
    {
        this->Input("geom")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("input_features")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output_features")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("pos_memo")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("batch_size").Int();
        this->Attr("num_points").Int();
        this->Attr("num_channels").Int();
        this->Attr("num_voxel_x").Int();
        this->Attr("num_voxel_y").Int();
        this->Attr("num_voxel_z").Int();

        this->SetInferShape(ge::InferShapeForVoxelPoolingTrain)
            .SetInferDataType(ge::InferDataTypeForVoxelPoolingTrain);
        this->AICore().SetTiling(optiling::TilingForVoxelPooling);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(VoxelPoolingTrain);
} // namespace ops