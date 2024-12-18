#include "roiaware_maxpool3d_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingForRoiawareMaxpool3dGrad(gert::TilingContext* context)
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
    uint32_t ONE_ELEMENT_SIZE = sizeof(float);
    constexpr uint32_t BYTE_ALIGNED = 32;
    constexpr float INPUT_UB_RATIO = 0.6;
    constexpr float OUTPUT_UB_RATIO = 0.2;
    constexpr uint32_t GRADOUT_BUFFER_NUMER = 3;
    constexpr uint32_t IDX_BUFFER_NUMBER = 2;
    constexpr uint32_t MAX_REPEAT_TIMES = 255;
    constexpr uint32_t COMPUTE_BYTE_SIZE = 256;
    uint64_t ubSize;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    if (context->GetInputShape(0) == nullptr || context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 0 : argmax, 1: gradOut, 2: gradIn
    auto argmaxShape = context->GetInputShape(0)->GetStorageShape();
    auto gradInShape = context->GetOutputShape(0)->GetStorageShape();

    uint32_t boxesNum = argmaxShape.GetDim(0);
    uint32_t outX = argmaxShape.GetDim(1);
    uint32_t outY = argmaxShape.GetDim(2);
    uint32_t outZ = argmaxShape.GetDim(3);
    uint32_t channels = argmaxShape.GetDim(4);
    uint32_t npoints = gradInShape.GetDim(0);

    uint32_t totalTask = boxesNum * outX * outY * outZ;
    uint32_t coreTask = (totalTask + aivNum - 1) / aivNum;
    uint32_t smallCoreTask = coreTask - 1;
    uint32_t firstSmallCoreIdx = totalTask % aivNum == 0 ? aivNum : totalTask % aivNum;
    uint32_t channelAligned = ((channels * ONE_ELEMENT_SIZE + BYTE_ALIGNED - 1) / BYTE_ALIGNED) * (BYTE_ALIGNED / ONE_ELEMENT_SIZE);

    uint32_t singleLoopTask = (ubSize * INPUT_UB_RATIO) / (GRADOUT_BUFFER_NUMER * ONE_ELEMENT_SIZE + IDX_BUFFER_NUMBER * sizeof(int32_t)) / channelAligned;
    uint32_t singleLoopOutput = (ubSize * OUTPUT_UB_RATIO) / ONE_ELEMENT_SIZE / 3 / channelAligned;

    uint32_t singleLoopTask_add = singleLoopTask <= MAX_REPEAT_TIMES? singleLoopTask : MAX_REPEAT_TIMES;   // for Add api
    uint32_t singleLoopTask_select = (COMPUTE_BYTE_SIZE * MAX_REPEAT_TIMES) / (channelAligned * ONE_ELEMENT_SIZE);  // for Select Api
    singleLoopTask = singleLoopTask_add < singleLoopTask_select? singleLoopTask_add : singleLoopTask_select;
    /* Set tilingData */
    RoiawareMaxpool3dGradTilingData tilingData;

    tilingData.set_totalTask(totalTask);
    tilingData.set_coreTask(coreTask);
    tilingData.set_firstSmallCoreIdx(firstSmallCoreIdx);
    tilingData.set_singleLoopTask(singleLoopTask);
    tilingData.set_singleLoopOutput(singleLoopOutput);
    tilingData.set_channelAligned(channelAligned);
    tilingData.set_channels(channels);
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
static ge::graphStatus InferShapeForRoiawareMaxpool3dGrad(gert::InferShapeContext* context)
{
    const gert::Shape* argmax = context->GetInputShape(0);
    const gert::Shape* gradOut = context->GetInputShape(1);
    gert::Shape* gradIn = context->GetOutputShape(0);
    if (argmax == nullptr || gradOut == nullptr || gradIn == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    int64_t channels = argmax->GetDim(4);
    auto runtimeAttrs = context->GetAttrs();
    const int32_t *npoints = (int32_t *)runtimeAttrs->GetInt(5);
    *gradIn = {*npoints, channels};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForRoiawareMaxpool3dGrad(gert::InferDataTypeContext* context)
{
    const ge::DataType value_dtype = context->GetInputDataType(1);
    context->SetOutputDataType(0, value_dtype);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class RoiawareMaxpool3dGrad : public OpDef {
public:
    explicit RoiawareMaxpool3dGrad(const char* name) : OpDef(name)
    {
        this->Input("argmax")
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

        this->SetInferShape(ge::InferShapeForRoiawareMaxpool3dGrad).SetInferDataType(ge::InferDataTypeForRoiawareMaxpool3dGrad);
        this->AICore().SetTiling(optiling::TilingForRoiawareMaxpool3dGrad);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(RoiawareMaxpool3dGrad);
}
