#include "graph_softmax_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
    constexpr float AVALIABLE_UB_RATIO = 0.8;
    constexpr uint32_t SRC_IDX = 0;   // number of edge
    constexpr uint32_t OUTPUT_IDX = 0;
    constexpr uint32_t EDGE_IDX = 0;
    constexpr uint32_t FEATURE_IDX = 1;
    constexpr uint32_t N_IDX = 0;
    constexpr uint32_t SINGLE_LOOP_TASK = 2360;   // 198*1024*0.8/4/(8*2+1),  UB*AVALIABLE_UB_RATIO*(size_of_float/int)/(8*2+1)
}   // some const express

namespace optiling {
    static ge::graphStatus TilingForGraphSoftmax(gert::TilingContext* context)
    {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        auto platformInfoPtr = context->GetPlatformInfo();
        if (platformInfoPtr == nullptr) {
            return ge::GRAPH_FAILED;
        }
        auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
        auto aivNum = ascendplatformInfo.GetCoreNumAiv();
        context->SetBlockDim(aivNum);
        uint64_t ubSize;
        ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        ubSize *= AVALIABLE_UB_RATIO;

        if (aivNum == 0 || ubSize == 0) {
            return ge::GRAPH_FAILED;
        }

        const gert::StorageShape *srcShape = context->GetInputShape(SRC_IDX); // [num_edge, num_feature]
        const gert::RuntimeAttrs *attr = context->GetAttrs();
        if (srcShape == nullptr || attr == nullptr) {
            return ge::GRAPH_FAILED;
        }
        auto NPtr = attr->GetAttrPointer<int>(N_IDX);
        if (NPtr == nullptr) {
            return ge::GRAPH_FAILED;
        }
        uint32_t N = *NPtr;
        uint32_t numEdge = srcShape->GetStorageShape().GetDim(EDGE_IDX);
        uint32_t numFeature = srcShape->GetStorageShape().GetDim(FEATURE_IDX);
        uint32_t totalTask = numEdge;
        uint32_t coreTask = (totalTask + aivNum - 1) / aivNum;
        uint32_t coreWorkspace = (N + aivNum - 1) / aivNum;
        uint32_t totalWorkspace = coreWorkspace * aivNum;
        uint32_t bigCoreCount = totalTask % aivNum == 0? aivNum : totalTask % aivNum;
        uint32_t singleLoopTaskCount = SINGLE_LOOP_TASK;

        GraphSoftmaxTilingData tilingData;

        tilingData.set_coreTask(coreTask);
        tilingData.set_coreWorkspace(coreWorkspace);
        tilingData.set_totalTask(totalTask);
        tilingData.set_totalWorkspace(totalWorkspace);
        tilingData.set_bigCoreCount(bigCoreCount);
        tilingData.set_singleLoopTaskCount(singleLoopTaskCount);

        if (context->GetRawTilingData() == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

        size_t systemWorkspaceSize = ascendplatformInfo.GetLibApiWorkSpaceSize();
        size_t usrWorkSpaceSize = 2 * totalWorkspace * numFeature * sizeof(float);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace == nullptr) {
            return ge::GRAPH_FAILED;
        }
        currentWorkspace[0] = systemWorkspaceSize + usrWorkSpaceSize;
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    static ge::graphStatus InferShapeForGraphSoftmax(gert::InferShapeContext* context)
    {
        const gert::Shape* src = context->GetInputShape(SRC_IDX);
        gert::Shape* softmaxResult = context->GetOutputShape(OUTPUT_IDX);
        if (src == nullptr || softmaxResult == nullptr) {
            return ge::GRAPH_FAILED;
        }
        uint64_t numEdge = src->GetDim(EDGE_IDX);
        uint64_t numFeature = src->GetDim(FEATURE_IDX);

        *softmaxResult = {numEdge, numFeature};
        return GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataTypeForGraphSoftmax(gert::InferDataTypeContext* context)
    {
        const ge::DataType num_valid_dtype = context->GetInputDataType(SRC_IDX);
        context->SetOutputDataType(OUTPUT_IDX, num_valid_dtype);
        return GRAPH_SUCCESS;
    }
}

namespace ops {
class GraphSoftmax : public OpDef {
public:
    explicit GraphSoftmax(const char* name) : OpDef(name)
    {
        this->Input("src")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("softmax_result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("N").Int();

        this->SetInferShape(ge::InferShapeForGraphSoftmax).SetInferDataType(ge::InferDataTypeForGraphSoftmax);
        this->AICore().SetTiling(optiling::TilingForGraphSoftmax);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GraphSoftmax);
}