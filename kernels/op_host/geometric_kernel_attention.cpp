
#include "geometric_kernel_attention_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ge/utils.h"

namespace {
const int32_t VALUE_PTR_INDEX = 0;
const int32_t SPATIAL_PTR_INDEX = 1;
const int32_t LEVEL_PTR_INDEX = 2;
const int32_t SAMPLING_PTR_INDEX = 3;
const int32_t WEIGHT_PTR_INDEX = 4;
const int32_t OUTPUT_PTR_INDEX = 0;
const int32_t BS_INDEX = 0;
const int32_t HEAD_INDEX = 1;
const int32_t KEY_INDEX = 2;
const int32_t DIM_INDEX = 3;
const int32_t LEVEL_INDEX = 0;
const int32_t QUERY_INDEX = 1;
const int32_t POINT_INDEX = 4;
const int32_t ALIGN_NUM = 8;
}

namespace optiling {
static ge::graphStatus TilingForGeometricKernelAttention(gert::TilingContext* context)
{
    GeometricKernelAttentionTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto valueTensorPtr = context->GetInputTensor(VALUE_PTR_INDEX);
    auto spatialshapeTensorPtr = context->GetInputTensor(SPATIAL_PTR_INDEX);
    auto levelstartindexTensorPtr = context->GetInputShape(LEVEL_PTR_INDEX);
    auto samplinglocationsTensorPtr = context->GetInputShape(SAMPLING_PTR_INDEX);
    auto attentionweightsTensorPtr = context->GetInputShape(WEIGHT_PTR_INDEX);
    if (valueTensorPtr == nullptr || spatialshapeTensorPtr == nullptr || levelstartindexTensorPtr == nullptr || samplinglocationsTensorPtr == nullptr || attentionweightsTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto ValueShape = context->GetInputShape(VALUE_PTR_INDEX);
    auto SpatialShape = context->GetInputShape(SPATIAL_PTR_INDEX);
    auto SamplingShape = context->GetInputShape(SAMPLING_PTR_INDEX);
    if (ValueShape == nullptr || SpatialShape == nullptr || SamplingShape == nullptr || context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int32_t batchSize = ValueShape->GetStorageShape().GetDim(BS_INDEX);
    int32_t numHeads = ValueShape->GetStorageShape().GetDim(HEAD_INDEX);
    int32_t numKeys = ValueShape->GetStorageShape().GetDim(KEY_INDEX);
    int32_t dim = ValueShape->GetStorageShape().GetDim(DIM_INDEX);
    int32_t numLevels = SpatialShape->GetStorageShape().GetDim(LEVEL_INDEX);
    int32_t numQueries = SamplingShape->GetStorageShape().GetDim(QUERY_INDEX);
    int32_t numPoints = SamplingShape->GetStorageShape().GetDim(POINT_INDEX);

    int32_t totalTaskNum = batchSize * numQueries * numHeads;
    int32_t alignTaskNum = AlignUp(totalTaskNum, ALIGN_NUM);
    int32_t alignLevels = AlignUp(numLevels, ALIGN_NUM);
    int32_t alignDim = AlignUp(dim, ALIGN_NUM);

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

    int32_t tailNum = alignTaskNum - totalTaskNum;
    uint32_t taskNumPerScore = (alignTaskNum / blockDim / ALIGN_NUM) * ALIGN_NUM;
    uint32_t taskNumPerLcore = taskNumPerScore + ALIGN_NUM;
    uint32_t scoreNum = (blockDim * (ALIGN_NUM + taskNumPerScore) - alignTaskNum) / ALIGN_NUM;
    uint32_t lcoreNum = blockDim - scoreNum;
      
    if (taskNumPerScore == 0) {
        blockDim = blockDim - scoreNum;
    }
    if (taskNumPerLcore == 0) {
        blockDim = blockDim - lcoreNum;
    }
    
    tiling.set_blockDim(blockDim);
    tiling.set_ubTotalSize(ubTotalSize);
    tiling.set_batchSize(batchSize);
    tiling.set_numKeys(numKeys);
    tiling.set_numHeads(numHeads);
    tiling.set_numQueries(numQueries);
    tiling.set_numLevels(numLevels);
    tiling.set_numPoints(numPoints);
    tiling.set_dim(dim);
    tiling.set_alignLevels(alignLevels);
    tiling.set_alignDim(alignDim);
    tiling.set_totalTaskNum(totalTaskNum);
    tiling.set_alignTaskNum(alignTaskNum);
    tiling.set_tailNum(tailNum);
    tiling.set_taskNumPerScore(taskNumPerScore);
    tiling.set_taskNumPerLcore(taskNumPerLcore);
    tiling.set_scoreNum(scoreNum);
    tiling.set_lcoreNum(lcoreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(blockDim);
    context->SetTilingKey(1);
    size_t systemWorkspaceSize = platformInfo.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShapeGeometricKernelAttention(gert::InferShapeContext* context)
{
    const gert::Shape* ValueShape = context->GetInputShape(VALUE_PTR_INDEX);
    const gert::Shape* SpatialShape = context->GetInputShape(SPATIAL_PTR_INDEX);
    const gert::Shape* LevelShape = context->GetInputShape(LEVEL_PTR_INDEX);
    const gert::Shape* SamplingShape = context->GetInputShape(SAMPLING_PTR_INDEX);
    const gert::Shape* AttentionShape = context->GetInputShape(WEIGHT_PTR_INDEX);
    gert::Shape* OutputShape = context->GetOutputShape(OUTPUT_PTR_INDEX);

    if (ValueShape == nullptr || SpatialShape == nullptr || LevelShape == nullptr || SamplingShape == nullptr || AttentionShape == nullptr || OutputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int32_t batchSize, numQueries, numHeads, dim;
    batchSize = ValueShape->GetDim(BS_INDEX);
    numHeads = ValueShape->GetDim(HEAD_INDEX);
    dim = ValueShape->GetDim(DIM_INDEX);
    numQueries = SamplingShape->GetDim(QUERY_INDEX);

    *OutputShape = {batchSize, numQueries, numHeads * dim};
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeGeometricKernelAttention(gert::InferDataTypeContext* context)
{
    const ge::DataType valueDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, valueDtype);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class GeometricKernelAttention : public OpDef {
public:
    explicit GeometricKernelAttention(const char* name) : OpDef(name)
    {
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("spatial_shapes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("level_start_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sampling_locations")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("attention_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeGeometricKernelAttention)
            .SetInferDataType(ge::InferDataTypeGeometricKernelAttention);

        this->AICore()
            .SetTiling(optiling::TilingForGeometricKernelAttention);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GeometricKernelAttention);
}
