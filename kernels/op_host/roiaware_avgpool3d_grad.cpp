#include "roiaware_avgpool3d_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingForRoiawareAvgpool3dGrad(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    auto aivNum = ascendplatformInfo.GetCoreNumAiv();
    context->SetBlockDim(aivNum);

    if (aivNum == 0) {
        return ge::GRAPH_FAILED;
    }
    constexpr uint32_t ONE_ELEMENT_SIZE = sizeof(float);
    constexpr uint32_t BYTE_ALIGNED = 32;
    constexpr float INPUT_UB_RATIO = 0.4;
    constexpr float OUTPUT_UB_RATIO = 0.4;
    constexpr uint32_t BUFFER_NUM = 2;

    uint64_t ubSize;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    if (context->GetInputShape(0) == nullptr ||
        context->GetInputShape(1) == nullptr ||
        context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 0 : argmax, 1: gradOut, 2: gradIn
    auto ptsIdxShape = context->GetInputShape(0)->GetStorageShape();
    auto gradOutShape = context->GetInputShape(1)->GetStorageShape();
    auto gradInShape = context->GetOutputShape(0)->GetStorageShape();

    uint32_t boxesNum = gradOutShape.GetDim(0);
    uint32_t outX = gradOutShape.GetDim(1);
    uint32_t outY = gradOutShape.GetDim(2);
    uint32_t outZ = gradOutShape.GetDim(3);
    uint32_t channels = gradOutShape.GetDim(4);
    uint32_t npoints = gradInShape.GetDim(0);
    uint32_t maxPtsPerVoxel = ptsIdxShape.GetDim(4) - 1;

    uint32_t totalTask = boxesNum * outX * outY * outZ;
    uint32_t singleCoreTask = (totalTask + aivNum - 1) / aivNum;
    uint32_t channelAligned = ((channels * ONE_ELEMENT_SIZE + BYTE_ALIGNED - 1) / BYTE_ALIGNED) * (BYTE_ALIGNED / ONE_ELEMENT_SIZE);
    uint32_t maxPtsPerVoxelAligned = ((maxPtsPerVoxel * sizeof(int32_t) + BYTE_ALIGNED - 1) / BYTE_ALIGNED) * (BYTE_ALIGNED / sizeof(int32_t));

    uint32_t singleLoopTask = (ubSize * INPUT_UB_RATIO) / BUFFER_NUM / (ONE_ELEMENT_SIZE * channelAligned + sizeof(int32_t) * maxPtsPerVoxelAligned);
    uint32_t outputLoopTask = (ubSize * OUTPUT_UB_RATIO) / (ONE_ELEMENT_SIZE * channelAligned);
    uint32_t bigCoreCount = totalTask % aivNum == 0? aivNum : totalTask % aivNum;

    /* Set tilingData */
    RoiawareAvgpool3dGradTilingData tilingData;
    tilingData.set_singleCoreTask(singleCoreTask);
    tilingData.set_singleLoopTask(singleLoopTask);
    tilingData.set_outputLoopTask(outputLoopTask);
    tilingData.set_channels(channels);
    tilingData.set_channelAligned(channelAligned);
    tilingData.set_maxPtsPerVoxel(maxPtsPerVoxel);
    tilingData.set_maxPtsPerVoxelAligned(maxPtsPerVoxelAligned);
    tilingData.set_bigCoreCount(bigCoreCount);
    tilingData.set_totalTask(totalTask);
    tilingData.set_npoints(npoints);
    
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t systemWorkspaceSize = ascendplatformInfo.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShapeForRoiawareAvgpool3dGrad(gert::InferShapeContext* context)
{
    const gert::Shape* ptsIdx = context->GetInputShape(0);
    const gert::Shape* gradOut = context->GetInputShape(1);
    gert::Shape* gradIn = context->GetOutputShape(0);
    if (ptsIdx == nullptr || gradOut == nullptr || gradIn == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    int64_t channels = gradOut->GetDim(4);
    auto runtimeAttrs = context->GetAttrs();
    const int32_t *npoints = (int32_t *)runtimeAttrs->GetInt(5);
    *gradIn = {*npoints, channels};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForRoiawareAvgpool3dGrad(gert::InferDataTypeContext* context)
{
    const ge::DataType value_dtype = context->GetInputDataType(1);
    context->SetOutputDataType(0, value_dtype);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class RoiawareAvgpool3dGrad : public OpDef {
public:
    explicit RoiawareAvgpool3dGrad(const char* name) : OpDef(name)
    {
        this->Input("pts_idx_of_voxels")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("grad_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("grad_in")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
            
        this->Attr("boxes_num").Int();
        this->Attr("out_x").Int();
        this->Attr("out_y").Int();
        this->Attr("out_z").Int();
        this->Attr("channels").Int();
        this->Attr("npoints").Int();
        this->Attr("max_pts_per_voxel").Int();

        this->SetInferShape(ge::InferShapeForRoiawareAvgpool3dGrad).SetInferDataType(ge::InferDataTypeForRoiawareAvgpool3dGrad);
        this->AICore().SetTiling(optiling::TilingForRoiawareAvgpool3dGrad);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(RoiawareAvgpool3dGrad);
}
