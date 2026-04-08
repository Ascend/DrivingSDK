#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "sparse_inverse_conv3d_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
using namespace ge;

namespace {
const uint32_t FEATURE_IDX = 0;
const uint32_t ORIGIN_INDICES_IDX = 1;
const uint32_t OUTPUT_FEATURE_IDX = 0;
const uint32_t INPUT_INDICES_OFFSET_IDX = 1;
const uint32_t ATTR_KERNELS_IDX = 0;
const uint32_t ATTR_IN_CHANNELS_IDX = 1;

const uint32_t TOTAL_TASK_DIM_IDX = 0;

const uint32_t KERNEL_SIZE_IDX_0 = 0;
const uint32_t KERNEL_SIZE_IDX_1 = 1;
const uint32_t KERNEL_SIZE_IDX_2 = 2;
const uint32_t KERNEL_INCHANNEL_IDX = 3;
const uint32_t NUM_BUFFER = 2; // double buffer
}; // namespace

namespace optiling {
static ge::graphStatus TilingForSparseInverseConv3d(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);

    // GetVectorTilingData
    uint32_t coreNum = ascendplatformInfo.GetCoreNumAiv();
    auto featureShapePtr = context->GetInputTensor(FEATURE_IDX);
    auto attrsPtr = context->GetAttrs();
    auto kernelSizePtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(ATTR_KERNELS_IDX);
    auto inChannelsPtr = attrsPtr->GetAttrPointer<int32_t>(ATTR_IN_CHANNELS_IDX);

    if (!featureShapePtr || !attrsPtr || !kernelSizePtr || !inChannelsPtr) {
        return ge::GRAPH_FAILED;
    }

    auto kernelSizeArr = reinterpret_cast<const int64_t*>(kernelSizePtr->GetData());
    int32_t kernelSize =
        kernelSizeArr[KERNEL_SIZE_IDX_0] * kernelSizeArr[KERNEL_SIZE_IDX_1] * kernelSizeArr[KERNEL_SIZE_IDX_2];
    int32_t inChannel = *inChannelsPtr;
    uint64_t availableUbSize;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, availableUbSize);
    uint32_t reserveUbSize = 1024;
    uint32_t indicesDataTypeSize = 4;
    uint32_t featureDataTypeSize = 4; //fp32
    uint32_t moveLen = (availableUbSize - reserveUbSize) /
                       (indicesDataTypeSize + indicesDataTypeSize * kernelSize + inChannel * featureDataTypeSize) /
                       NUM_BUFFER;
    uint32_t totalTaskCount = featureShapePtr->GetStorageShape().GetDim(TOTAL_TASK_DIM_IDX);
    uint32_t vectorCoreTask = Ceil(totalTaskCount, coreNum);
    uint32_t usedVectorCoreNum = Ceil(totalTaskCount, vectorCoreTask);
    uint32_t vectorLastCoreTask = Tail(totalTaskCount, vectorCoreTask);

    uint32_t coreRepeatTimes = Ceil(vectorCoreTask, moveLen);
    uint32_t lastCoreRepeatTimes = Ceil(vectorLastCoreTask, moveLen);
    uint32_t coreMoveLenTail = Tail(vectorCoreTask, moveLen);
    uint32_t lastCoreMoveLenTail = Tail(vectorLastCoreTask, moveLen);

    // SetTilingData
    SparseInverseConv3dTilingData tilingData;
    context->SetBlockDim(usedVectorCoreNum);
    tilingData.set_inChannel(inChannel);
    tilingData.set_kernelSize(kernelSize);
    tilingData.set_vectorCoreTask(vectorCoreTask);
    tilingData.set_vectorLastCoreTask(vectorLastCoreTask);
    tilingData.set_moveLen(moveLen);
    tilingData.set_coreRepeatTimes(coreRepeatTimes);
    tilingData.set_coreMoveLenTail(coreMoveLenTail);
    tilingData.set_lastCoreRepeatTimes(lastCoreRepeatTimes);
    tilingData.set_lastCoreMoveLenTail(lastCoreMoveLenTail);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 1;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForSparseInverseConv3d(gert::InferShapeContext* context)
{
    const gert::Shape* indicesShape = context->GetInputShape(ORIGIN_INDICES_IDX);
    if (indicesShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto totalTaskCount = indicesShape->GetDim(TOTAL_TASK_DIM_IDX);
    auto kernelSizePtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(ATTR_KERNELS_IDX);
    auto kernelSizeData = reinterpret_cast<const int64_t*>(kernelSizePtr->GetData());
    uint32_t kernelD = kernelSizeData[KERNEL_SIZE_IDX_0];
    uint32_t kernelH = kernelSizeData[KERNEL_SIZE_IDX_1];
    uint32_t kernelW = kernelSizeData[KERNEL_SIZE_IDX_2];
    uint32_t kernelSize = kernelD * kernelH * kernelW;
    auto inChannelsPtr = attrsPtr->GetAttrPointer<int32_t>(ATTR_IN_CHANNELS_IDX);

    gert::Shape* outFeatureShape = context->GetOutputShape(OUTPUT_FEATURE_IDX);
    if (outFeatureShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outFeatureShape->SetDimNum(0);
    outFeatureShape->AppendDim(totalTaskCount);
    outFeatureShape->AppendDim(*inChannelsPtr * kernelSize);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeForSparseInverseConv3d(gert::InferDataTypeContext* context)
{
    const ge::DataType feature_dtype = context->GetInputDataType(FEATURE_IDX);
    context->SetOutputDataType(0, feature_dtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class SparseInverseConv3d : public OpDef {
public:
    explicit SparseInverseConv3d(const char* name) : OpDef(name)
    {
        this->Input("features") // origin_features
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("indices") // origin_indices
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("unique_indices_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("sorted_idx_to_former_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("output_img2col")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("kernel_size").AttrType(REQUIRED).ListInt();
        this->Attr("in_channels").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShapeForSparseInverseConv3d)
            .SetInferDataType(ge::InferDtypeForSparseInverseConv3d);
        this->AICore().SetTiling(optiling::TilingForSparseInverseConv3d);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SparseInverseConv3d);
} // namespace ops
