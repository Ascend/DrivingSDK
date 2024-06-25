/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */

#include "voxel_pooling_train_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

constexpr uint32_t FP32_MODE = 0;
constexpr uint32_t FP16_MODE = 1;
constexpr uint32_t WORKSPACE_16MBYTE_SIZE = 16 * 1024 * 1024;

namespace optiling {
static ge::graphStatus TilingForVoxelPooling(gert::TilingContext* context)
{
    VoxelPoolingTilingData tiling;
    // get core num
    auto platformInfoPtr = context->GetPlatformInfo();
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
    int batchSize = *(attrsPtr->GetAttrPointer<int>(0));
    int numPoints = *(attrsPtr->GetAttrPointer<int>(1));
    int numChannels = *(attrsPtr->GetAttrPointer<int>(2));
    int numVoxelX = *(attrsPtr->GetAttrPointer<int>(3));
    int numVoxelY = *(attrsPtr->GetAttrPointer<int>(4));
    int numVoxelZ = *(attrsPtr->GetAttrPointer<int>(5));

    // set workspace
    size_t sysWorkspaceSize = WORKSPACE_16MBYTE_SIZE + batchSize * numVoxelX * numVoxelY * numChannels * sizeof(float);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;

    uint32_t featuresNumInCore = numPoints / coreNum;
    uint32_t featuresNumInLastCore = numPoints - featuresNumInCore * (coreNum - 1);

    // save param
    context->SetBlockDim(coreNum);
    tiling.set_core_num(coreNum);
    tiling.set_features_num_in_core(featuresNumInCore);
    tiling.set_features_num_in_last_core(featuresNumInLastCore);
    tiling.set_batch_size(batchSize);
    tiling.set_num_points(numPoints);
    tiling.set_num_channels(numChannels);
    tiling.set_num_voxel_x(numVoxelX);
    tiling.set_num_voxel_y(numVoxelY);
    tiling.set_num_voxel_z(numVoxelZ);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
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
    if (attrsPtr == nullptr || outFeaturesShape == nullptr ||
        posMemoShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int batchSize = *(attrsPtr->GetAttrPointer<int>(0));
    int numPoints = *(attrsPtr->GetAttrPointer<int>(1));
    int numChannels = *(attrsPtr->GetAttrPointer<int>(2));
    int numVoxelX = *(attrsPtr->GetAttrPointer<int>(3));
    int numVoxelY = *(attrsPtr->GetAttrPointer<int>(4));

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

        this->SetInferShape(ge::InferShapeForVoxelPoolingTrain);
        this->AICore().SetTiling(optiling::TilingForVoxelPooling);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910c");
    }
};
OP_ADD(VoxelPoolingTrain);
} // namespace ops