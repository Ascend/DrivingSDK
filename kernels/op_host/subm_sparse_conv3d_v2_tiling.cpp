#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "common.h"
#include "subm_sparse_conv3d_v2_tiling.h"
using namespace ge;
using namespace std;
using namespace AscendC;

namespace {
const uint32_t INPUT_FEATURE_IDX = 0;
const uint32_t INPUT_INDICES_IDX = 1;
const uint32_t OUTPUT_FEATURE_IDX = 0;
const uint32_t INPUT_INDICES_OFFSET_IDX = 1;
const uint32_t ATTR_KERNELS_IDX = 0;
const uint32_t ATTR_IN_CHANNELS_IDX = 1;
const uint32_t ATTR_SPATIAL_SHAPE_IDX = 2;
const uint32_t ATTR_BATCH_SIZE_IDX = 3;
const uint32_t TOTAL_TASK_DIM_IDX = 0;

const uint32_t KERNEL_SIZE_IDX_0 = 0;
const uint32_t KERNEL_SIZE_IDX_1 = 1;
const uint32_t KERNEL_SIZE_IDX_2 = 2;

const uint32_t OUT_SPATIAL_SHAPE_IDX_0 = 0;
const uint32_t OUT_SPATIAL_SHAPE_IDX_1 = 1;
const uint32_t OUT_SPATIAL_SHAPE_IDX_2 = 2;

const int32_t BYTE_ALIGN_SIZE = 32;
const int32_t FLOAT_BYTE_SIZE = 4;
const float AVALIABLE_UB_RATIO = 0.7;
const float SINGLE_LOOP_UB_SIZE = 8 * 4;
};


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SubmSparseConv3dV2TilingData tiling;
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
    context->SetBlockDim(aivNum);

    auto attrsPtr = context->GetAttrs();
    if (aivNum == 0 || context->GetInputTensor(INPUT_FEATURE_IDX) == nullptr || attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto featureShapeArr = context->GetInputTensor(INPUT_FEATURE_IDX)->GetStorageShape();
    auto kernelSizePtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(ATTR_KERNELS_IDX);
    auto outSpatialShapePtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(ATTR_SPATIAL_SHAPE_IDX);
    auto inChannelsPtr = attrsPtr->GetAttrPointer<int32_t>(ATTR_IN_CHANNELS_IDX);
    auto batchSizePtr = attrsPtr->GetAttrPointer<int32_t>(ATTR_BATCH_SIZE_IDX);
    if (kernelSizePtr == nullptr || outSpatialShapePtr == nullptr || inChannelsPtr == nullptr || batchSizePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto kernelSizeArr = reinterpret_cast<const int64_t*>(kernelSizePtr->GetData());
    auto outSpatialShapeArr = reinterpret_cast<const int64_t*>(outSpatialShapePtr->GetData());
    uint32_t totalTaskCount = featureShapeArr.GetDim(TOTAL_TASK_DIM_IDX);
    uint32_t coreTaskCount = totalTaskCount / aivNum;
    uint32_t bigCoreCount = totalTaskCount % aivNum;
    uint32_t singleLoopTask = ubSize / (SINGLE_LOOP_UB_SIZE +
        CeilAlign(*inChannelsPtr, BYTE_ALIGN_SIZE / FLOAT_BYTE_SIZE) * FLOAT_BYTE_SIZE);

    tiling.set_k0(kernelSizeArr[KERNEL_SIZE_IDX_0]);
    tiling.set_k1(kernelSizeArr[KERNEL_SIZE_IDX_1]);
    tiling.set_k2(kernelSizeArr[KERNEL_SIZE_IDX_2]);
    tiling.set_spatialShape0(outSpatialShapeArr[OUT_SPATIAL_SHAPE_IDX_0]);
    tiling.set_spatialShape1(outSpatialShapeArr[OUT_SPATIAL_SHAPE_IDX_1]);
    tiling.set_spatialShape2(outSpatialShapeArr[OUT_SPATIAL_SHAPE_IDX_2]);

    tiling.set_batchSize(*batchSizePtr);
    tiling.set_inChannels(*inChannelsPtr);
    tiling.set_coreTaskCount(coreTaskCount);
    tiling.set_bigCoreCount(bigCoreCount);
    tiling.set_singleLoopTask(singleLoopTask);
    
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 1;
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
    auto batchSize = *(attrsPtr->GetAttrPointer<int32_t>(ATTR_BATCH_SIZE_IDX));
    auto inChannels = *(attrsPtr->GetAttrPointer<int32_t>(ATTR_IN_CHANNELS_IDX));

    outFeatureShape->SetDimNum(0);
    outFeatureShape->AppendDim(totalTaskCount);
    outFeatureShape->AppendDim(inChannels * kernelDataSize);
    indicesOffsetShape->SetDimNum(0);
    indicesOffsetShape->AppendDim(outputDataSize);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class SubmSparseConv3dV2 : public OpDef {
public:
    explicit SubmSparseConv3dV2(const char* name) : OpDef(name)
    {
        this->Input("feature")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("map1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("map2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("feature_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("indices_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("kernel_size")
            .AttrType(REQUIRED)
            .ListInt();
        this->Attr("in_channels")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("out_spatial_shape")
            .AttrType(REQUIRED)
            .ListInt();
        this->Attr("batch_size")
            .AttrType(REQUIRED)
            .Int();

        this->SetInferShape(ge::InferShape);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SubmSparseConv3dV2);
}