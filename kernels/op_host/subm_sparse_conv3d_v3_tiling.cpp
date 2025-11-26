#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "common.h"
#include "subm_sparse_conv3d_v3_tiling.h"
using namespace ge;
using namespace std;
using namespace AscendC;

namespace {
const uint32_t INPUT_FEATURE_IDX = 0;
const uint32_t INPUT_WEIGHT_IDX = 1;
const uint32_t INPUT_INDICES_IDX = 2;
const uint32_t OUTPUT_FEATURE_IDX = 0;
const uint32_t INPUT_INDICES_OFFSET_IDX = 1;

const uint32_t ATTR_KERNELS_IDX = 0;
const uint32_t ATTR_IN_CHANNELS_IDX = 1;
const uint32_t ATTR_OUT_CHANNELS_IDX = 2;
const uint32_t ATTR_SPATIAL_SHAPE_IDX = 3;
const uint32_t ATTR_BATCH_SIZE_IDX = 4;
const uint32_t ATTR_WITH_KEY_IDX = 5;

const uint32_t TOTAL_TASK_DIM_IDX = 0;

const uint32_t KERNEL_SIZE_IDX_0 = 0;
const uint32_t KERNEL_SIZE_IDX_1 = 1;
const uint32_t KERNEL_SIZE_IDX_2 = 2;

const uint32_t OUT_SPATIAL_SHAPE_IDX_0 = 0;
const uint32_t OUT_SPATIAL_SHAPE_IDX_1 = 1;
const uint32_t OUT_SPATIAL_SHAPE_IDX_2 = 2;

const int32_t BYTE_ALIGN_SIZE = 32;
const float AVALIABLE_UB_RATIO = 0.8;
const float STAGE2_UB_RATIO = 0.2;
const int32_t INT32_BYTE_SIZE = 4;
const int32_t FLOAT_BYTE_SIZE = 4;
const int32_t HALF_BYTE_SIZE = 2;
const int32_t INDICES_BUFFER_LENGTH = 8;

};


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SubmSparseConv3dV3TilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    
    uint64_t ubSize;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize *= AVALIABLE_UB_RATIO;

    auto aivNum = ascendplatformInfo.GetCoreNumAiv();
    auto aicNum = ascendplatformInfo.GetCoreNumAic();
    context->SetBlockDim(aicNum);

    auto attrsPtr = context->GetAttrs();
    if (aivNum == 0 || context->GetInputTensor(INPUT_FEATURE_IDX) == nullptr || attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto featureShapeArr = context->GetInputTensor(INPUT_FEATURE_IDX)->GetStorageShape();
    auto kernelSizePtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(ATTR_KERNELS_IDX);
    auto outSpatialShapePtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(ATTR_SPATIAL_SHAPE_IDX);
    auto inChannelsPtr = attrsPtr->GetAttrPointer<int32_t>(ATTR_IN_CHANNELS_IDX);
    auto outChannelsPtr = attrsPtr->GetAttrPointer<int32_t>(ATTR_OUT_CHANNELS_IDX);
    auto batchSizePtr = attrsPtr->GetAttrPointer<int32_t>(ATTR_BATCH_SIZE_IDX);
    auto withKeyPtr = attrsPtr->GetAttrPointer<int32_t>(ATTR_WITH_KEY_IDX);
    auto featureDataTypePtr = context->GetInputDesc(INPUT_FEATURE_IDX);
    if (kernelSizePtr == nullptr || outSpatialShapePtr == nullptr || inChannelsPtr == nullptr || withKeyPtr == nullptr ||
        batchSizePtr == nullptr || outChannelsPtr == nullptr || featureDataTypePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto featureDataType = featureDataTypePtr->GetDataType();
    int32_t byteSizePerElements = featureDataType == ge::DT_FLOAT16? HALF_BYTE_SIZE : FLOAT_BYTE_SIZE;

    auto kernelSizeArr = reinterpret_cast<const int64_t*>(kernelSizePtr->GetData());
    auto outSpatialShapeArr = reinterpret_cast<const int64_t*>(outSpatialShapePtr->GetData());
    int32_t k0 = kernelSizeArr[KERNEL_SIZE_IDX_0];
    int32_t k1 = kernelSizeArr[KERNEL_SIZE_IDX_1];
    int32_t k2 = kernelSizeArr[KERNEL_SIZE_IDX_2];
    int32_t kernelSize = k0 * k1 * k2;
    int32_t withKey = *withKeyPtr;

    uint32_t mapValBufSize = CeilAlign(k0 * k1 * CeilAlign(k2, BYTE_ALIGN_SIZE / INT32_BYTE_SIZE), BYTE_ALIGN_SIZE / INT32_BYTE_SIZE);
    uint32_t inChannelAligned = CeilAlign(*inChannelsPtr, BYTE_ALIGN_SIZE / byteSizePerElements);
    uint32_t outChannelAligned = CeilAlign(*outChannelsPtr, BYTE_ALIGN_SIZE / byteSizePerElements);
    int32_t kernelSizeAligned = CeilAlign(kernelSize, BYTE_ALIGN_SIZE / byteSizePerElements);

    uint32_t totalTaskCount = featureShapeArr.GetDim(TOTAL_TASK_DIM_IDX);
    uint32_t coreTaskCount = totalTaskCount / aivNum;
    uint32_t bigCoreCount = totalTaskCount % aivNum;
    
    uint32_t singleLoopTask = FloorAlign(ubSize / (kernelSizeAligned * 2 + INDICES_BUFFER_LENGTH + mapValBufSize) / INT32_BYTE_SIZE, static_cast<uint64_t>(BYTE_ALIGN_SIZE / INT32_BYTE_SIZE));
    
    uint32_t stage2SingleLoopTask = (ubSize * STAGE2_UB_RATIO) / 2 / INT32_BYTE_SIZE;
    stage2SingleLoopTask = stage2SingleLoopTask > coreTaskCount + 1 ? coreTaskCount + 1 : stage2SingleLoopTask;
    stage2SingleLoopTask = CeilAlign(stage2SingleLoopTask, 64u);  // for CompareScalar

    uint32_t featureBufLen = (ubSize - 2 * stage2SingleLoopTask * INT32_BYTE_SIZE) /
        (inChannelAligned + outChannelAligned) / byteSizePerElements / 2;
    
    auto dataType = byteSizePerElements == FLOAT_BYTE_SIZE? matmul_tiling::DataType::DT_FLOAT : matmul_tiling::DataType::DT_FLOAT16;
    matmul_tiling::MatmulApiTiling mm0Tiling(ascendplatformInfo);
    mm0Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    mm0Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    mm0Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    mm0Tiling.SetOrgShape(coreTaskCount == 0 ? 1 : coreTaskCount, *outChannelsPtr, *inChannelsPtr);
    mm0Tiling.SetShape(coreTaskCount == 0 ? 1 : coreTaskCount, *outChannelsPtr, *inChannelsPtr);
    mm0Tiling.SetBias(false);
    mm0Tiling.SetBufferSpace(-1, -1, -1);
    if (mm0Tiling.GetTiling(tiling.mm0TilingData) == -1) {
        return ge::GRAPH_FAILED;
    }
    
    tiling.set_k0(k0);
    tiling.set_k1(k1);
    tiling.set_k2(k2);
    tiling.set_spatialShape0(outSpatialShapeArr[OUT_SPATIAL_SHAPE_IDX_0]);
    tiling.set_spatialShape1(outSpatialShapeArr[OUT_SPATIAL_SHAPE_IDX_1]);
    tiling.set_spatialShape2(outSpatialShapeArr[OUT_SPATIAL_SHAPE_IDX_2]);
    tiling.set_batchSize(*batchSizePtr);
    tiling.set_inChannels(*inChannelsPtr);
    tiling.set_outChannels(*outChannelsPtr);
    tiling.set_coreTaskCount(coreTaskCount);
    tiling.set_bigCoreCount(bigCoreCount);
    tiling.set_singleLoopTask(singleLoopTask);
    tiling.set_totalTaskCount(totalTaskCount);
    tiling.set_availableUBSize(ubSize);
    tiling.set_featureBufLen(featureBufLen);
    tiling.set_stage2SingleLoopTask(stage2SingleLoopTask);
    tiling.set_withKey(withKey);

    ADD_TILING_DATA(context, tiling);

    size_t systemWorkspaceSize = ascendplatformInfo.GetLibApiWorkSpaceSize();
    size_t usrWorkSpaceSize = 2 * aivNum * stage2SingleLoopTask * (*inChannelsPtr + *outChannelsPtr) * byteSizePerElements;

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    CHECK_NULLPTR(currentWorkspace);
    currentWorkspace[0] = systemWorkspaceSize + usrWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto kernelSizePtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(ATTR_KERNELS_IDX);
    auto kernelSizeArr = reinterpret_cast<const int64_t*>(kernelSizePtr->GetData());
    const gert::Shape* indicesShape = context->GetInputShape(INPUT_INDICES_IDX);
    gert::Shape* outFeatureShape = context->GetOutputShape(OUTPUT_FEATURE_IDX);
    if (outFeatureShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* indicesOffsetShape = context->GetOutputShape(INPUT_INDICES_OFFSET_IDX);
    if (indicesOffsetShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto kernelDataSize = kernelSizeArr[0] * kernelSizeArr[1] * kernelSizeArr[2];
    auto totalTaskCount = indicesShape->GetDim(TOTAL_TASK_DIM_IDX);
    auto outputDataSize = totalTaskCount * kernelDataSize;
    auto outChannels = *(attrsPtr->GetAttrPointer<int32_t>(ATTR_OUT_CHANNELS_IDX));

    outFeatureShape->SetDimNum(0);
    outFeatureShape->AppendDim(totalTaskCount);
    outFeatureShape->AppendDim(outChannels);
    indicesOffsetShape->SetDimNum(0);
    indicesOffsetShape->AppendDim(outputDataSize);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class SubmSparseConv3dV3 : public OpDef {
public:
    explicit SubmSparseConv3dV3(const char* name) : OpDef(name)
    {
        this->Input("feature")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("indices_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("map1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("map2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("feature_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out_indices_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("kernel_size")
            .AttrType(REQUIRED)
            .ListInt();
        this->Attr("in_channels")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("out_channels")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("out_spatial_shape")
            .AttrType(REQUIRED)
            .ListInt();
        this->Attr("batch_size")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("with_key")
            .AttrType(REQUIRED)
            .Int();

        this->SetInferShape(ge::InferShape);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SubmSparseConv3dV3);
}