#include "diff_iou_rotated_sort_vertices_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr float AVALIABLE_UB_RATIO = 0.6;
constexpr uint32_t VERTICES_IDX = 0;
constexpr uint32_t OUTPUT_IDX = 0;
constexpr uint32_t BATCH_SIZE_IDX = 0;
constexpr uint32_t NUM_BOXES_IDX = 1;
constexpr uint32_t OUTPUT_VERTICES_COUNT = 9;
constexpr uint32_t NUM_VALID_IDX = 2;
constexpr uint32_t SINGLE_LOOP_TASK = 64;
}   // some const express

namespace optiling {
static ge::graphStatus TilingForDiffIouRotatedSortVertices(gert::TilingContext* context)
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

    if (context->GetInputShape(VERTICES_IDX) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto verticesShape = context->GetInputShape(VERTICES_IDX)->GetStorageShape();
    uint32_t batchSize = verticesShape.GetDim(BATCH_SIZE_IDX);
    uint32_t numBoxes = verticesShape.GetDim(NUM_BOXES_IDX);
    
    uint32_t totalTask = batchSize * numBoxes;
    uint32_t coreTask = (totalTask + aivNum - 1) / aivNum;
    uint32_t bigCoreCount = totalTask % aivNum == 0? aivNum : totalTask % aivNum;
    uint32_t singleLoopTaskCount = SINGLE_LOOP_TASK;

    DiffIouRotatedSortVerticesTilingData tilingData;

    tilingData.set_coreTask(coreTask);
    tilingData.set_bigCoreCount(bigCoreCount);
    tilingData.set_singleLoopTaskCount(singleLoopTaskCount);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t systemWorkspaceSize = ascendplatformInfo.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShapeForDiffIouRotatedSortVertices(gert::InferShapeContext* context)
{
    const gert::Shape* vertices = context->GetInputShape(VERTICES_IDX);
    gert::Shape* sortedIdx = context->GetOutputShape(OUTPUT_IDX);
    if (vertices == nullptr || sortedIdx == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t batchSize = vertices->GetDim(BATCH_SIZE_IDX);
    int64_t numBoxes = vertices->GetDim(NUM_BOXES_IDX);

    *sortedIdx = {batchSize, numBoxes, OUTPUT_VERTICES_COUNT};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForDiffIouRotatedSortVertices(gert::InferDataTypeContext* context)
{
    const ge::DataType num_valid_dtype = context->GetInputDataType(NUM_VALID_IDX);
    context->SetOutputDataType(OUTPUT_IDX, num_valid_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class DiffIouRotatedSortVertices : public OpDef {
public:
    explicit DiffIouRotatedSortVertices(const char* name) : OpDef(name)
    {
        this->Input("vertices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("num_valid")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        
        this->Output("sortedIdx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForDiffIouRotatedSortVertices).SetInferDataType(ge::InferDataTypeForDiffIouRotatedSortVertices);
        this->AICore().SetTiling(optiling::TilingForDiffIouRotatedSortVertices);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(DiffIouRotatedSortVertices);
}