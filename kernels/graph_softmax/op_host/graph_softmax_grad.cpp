#include "graph_softmax_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ge/utils.h"

namespace {
const uint32_t NODE_NUM_INDEX = 0;
const uint32_t MAX_N_INDEX = 0;
const uint32_t INDEX_PTR_INDEX = 0;
const uint32_t SOFTMAX_OUTPUT_PTR_INDEX = 1;
const uint32_t GRAD_OUTPUT_PTR_INDEX = 2;
const uint32_t REDUCE_SUM_PTR_INDEX = 3;
const uint32_t SRC_GRAD_PTR_INDEX = 0;
const uint32_t EDGE_INDEX = 0;
const uint32_t ALIGN_NUM = 8;
const int32_t SRC_SHAPE_DIM = 8;
const int32_t FLOAT_SIZE = 4;
const int32_t BUFFER_NUM = 5;
const int32_t DOUBLE_NUM = 2;
const float UB_RATIO = 0.8f;
}

namespace optiling {
static ge::graphStatus TilingForGraphSoftmaxGrad(gert::TilingContext* context)
{
    GraphSoftmaxGradTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto indexTensorPtr = context->GetInputTensor(INDEX_PTR_INDEX);
    auto softmaxOutputTensorPtr = context->GetInputShape(SOFTMAX_OUTPUT_PTR_INDEX);
    auto gradOutputTensorPtr = context->GetInputShape(GRAD_OUTPUT_PTR_INDEX);
    auto reduceSumTensorPtr = context->GetInputShape(REDUCE_SUM_PTR_INDEX);
    if (indexTensorPtr == nullptr || softmaxOutputTensorPtr == nullptr || gradOutputTensorPtr == nullptr || reduceSumTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto indexShape = context->GetInputShape(INDEX_PTR_INDEX);
    auto softmaxOutputShape = context->GetInputShape(SOFTMAX_OUTPUT_PTR_INDEX);
    auto gradOutputShape = context->GetInputShape(GRAD_OUTPUT_PTR_INDEX);
    auto reduceSumShape = context->GetInputShape(REDUCE_SUM_PTR_INDEX);
    if (indexShape == nullptr || softmaxOutputShape == nullptr || gradOutputShape == nullptr || reduceSumShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int32_t nodeNum = *(attrsPtr->GetAttrPointer<int32_t>(NODE_NUM_INDEX));
    int32_t edgeNum = softmaxOutputShape->GetStorageShape().GetDim(EDGE_INDEX);

    auto platform = context->GetPlatformInfo();
    if (platform == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = platform_ascendc::PlatformAscendC(platform);
    uint64_t ubTotalSize;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubTotalSize);
    uint32_t blockDim = platformInfo.GetCoreNumAiv();
    if (blockDim == 0) {
        return ge::GRAPH_FAILED;
    }

    int32_t alignTaskNum = AlignUp(edgeNum, blockDim);
    int32_t tailNum = alignTaskNum - edgeNum;
    uint32_t taskNumPerCore = alignTaskNum / blockDim;

    int32_t tailCoreNum = blockDim - tailNum;
    int32_t taskNumPerLoop = static_cast<int32_t>((ubTotalSize * UB_RATIO) / (SRC_SHAPE_DIM * FLOAT_SIZE * BUFFER_NUM));
    taskNumPerLoop = static_cast<int32_t>((taskNumPerLoop + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM);
    int32_t taskLoop = static_cast<int32_t>((taskNumPerCore + taskNumPerLoop - 1) / taskNumPerLoop);
    if (taskNumPerCore <= taskNumPerLoop) {
        taskNumPerLoop = taskNumPerCore;
        taskLoop = 1;
    }

    tiling.set_edgeNum(edgeNum);
    tiling.set_alignTaskNum(alignTaskNum);
    tiling.set_tailNum(tailNum);
    tiling.set_nodeNum(nodeNum);
    tiling.set_tailCoreNum(tailCoreNum);
    tiling.set_taskNumPerCore(taskNumPerCore);
    tiling.set_taskNumPerLoop(taskNumPerLoop);
    tiling.set_taskLoop(taskLoop);
    tiling.set_blockDim(blockDim);
    tiling.set_ubTotalSize(ubTotalSize);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(blockDim);
    context->SetTilingKey(1);
    size_t systemWorkspaceSize = platformInfo.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShapeGraphSoftmaxGrad(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* indexShape = context->GetInputShape(INDEX_PTR_INDEX);
    const gert::Shape* softmaxOutputShape = context->GetInputShape(SOFTMAX_OUTPUT_PTR_INDEX);
    const gert::Shape* gradOutputShape = context->GetInputShape(GRAD_OUTPUT_PTR_INDEX);
    const gert::Shape* reduceSumShape = context->GetInputShape(REDUCE_SUM_PTR_INDEX);
    gert::Shape* srcGradShape = context->GetOutputShape(SRC_GRAD_PTR_INDEX);

    if (indexShape == nullptr || softmaxOutputShape == nullptr || gradOutputShape == nullptr || reduceSumShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int32_t edgeNum = softmaxOutputShape->GetDim(EDGE_INDEX);
    *srcGradShape = {edgeNum, SRC_SHAPE_DIM};
    return GRAPH_SUCCESS;
}


static ge::graphStatus InferDataTypeGraphSoftmaxGrad(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType valueDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, valueDtype);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class GraphSoftmaxGrad : public OpDef {
public:
    explicit GraphSoftmaxGrad(const char* name) : OpDef(name)
    {
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("softmax_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("grad_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("reduce_sum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("node_num").AttrType(REQUIRED).Int();
        this->Output("src_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeGraphSoftmaxGrad)
            .SetInferDataType(ge::InferDataTypeGraphSoftmaxGrad);

        this->AICore()
            .SetTiling(optiling::TilingForGraphSoftmaxGrad);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GraphSoftmaxGrad);
}
