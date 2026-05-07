#include <cstdint>
#include "ge/utils.h"
#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
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
const int32_t STAGE2_COPY_BUF_COUNT = 8;
const int32_t INT32_BYTE_SIZE = 4;
const int32_t FP32_BYTE_SIZE = 4;
const int32_t HALF_BYTE_SIZE = 2;
const int32_t INDICES_BUFFER_LENGTH = 8;

const int32_t GATHER_BUF_LEN = 4;
const int32_t SCATTER_BUF_LEN = 4;

const int32_t MIN_SINGLE_LOOP_TASK_COUNT = 64;
const int32_t MAX_SINGLE_LOOP_TASK_COUNT = 2048;

const int32_t LOCAL_MEM_SIZE = 216 * 1024;
};

namespace optiling {

uint32_t  setTiling(gert::TilingContext* context, SubmSparseConv3dV3TilingData& tiling, matmul_tiling::MatmulApiTiling& mmTiling, uint64_t ubSize,
    uint64_t totalTaskCount, uint32_t inChannels, uint32_t outChannels, int32_t byteSizePerElements, int32_t aivNum, int32_t k0, int32_t k1, int32_t k2,
    int32_t batchSize, int32_t withKey, int32_t spatialShape0, int32_t spatialShape1, int32_t spatialShape2)
{
    if (byteSizePerElements == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t elementsCountPerBlock = BYTE_ALIGN_SIZE / byteSizePerElements;
    uint32_t kernelSize = k0 * k1 * k2;
    uint32_t inChannelAligned = AlignUp(inChannels, elementsCountPerBlock);
    uint32_t outChannelAligned = AlignUp(outChannels, static_cast<uint32_t>(BYTE_ALIGN_SIZE / FP32_BYTE_SIZE));

    if (aivNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t coreTaskCount = (totalTaskCount + aivNum - 1) / aivNum;
    uint32_t bigCoreCount = totalTaskCount % aivNum;

    uint32_t singleLoopTask = AlignUp(coreTaskCount / 4, static_cast<uint32_t>(64));
    singleLoopTask = singleLoopTask > MIN_SINGLE_LOOP_TASK_COUNT? singleLoopTask : MIN_SINGLE_LOOP_TASK_COUNT;
    singleLoopTask = singleLoopTask < MAX_SINGLE_LOOP_TASK_COUNT? singleLoopTask : MAX_SINGLE_LOOP_TASK_COUNT;
    singleLoopTask = min(coreTaskCount, singleLoopTask);
    uint32_t singleLoopTaskAligned = AlignUp(singleLoopTask, static_cast<uint32_t>(BYTE_ALIGN_SIZE / INT32_BYTE_SIZE));

    uint32_t gatherBufLen = (ubSize - singleLoopTaskAligned * 3 * INT32_BYTE_SIZE - 32) / 4 / (inChannelAligned * byteSizePerElements);  // 对称性 + double buffer
    uint32_t scatterBufLen = (ubSize - singleLoopTaskAligned * 3 * INT32_BYTE_SIZE - 32) / 4 / (outChannelAligned * FP32_BYTE_SIZE);  // 对称性 + double buffer

    auto dataType = byteSizePerElements == FP32_BYTE_SIZE? matmul_tiling::DataType::DT_FLOAT : matmul_tiling::DataType::DT_FLOAT16;
    
    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mmTiling.SetOrgShape(singleLoopTask, outChannels, inChannels);
    mmTiling.SetShape(singleLoopTask, outChannels, inChannels);
    mmTiling.SetBias(false);
    mmTiling.SetBufferSpace(-1, -1, -1);

    mmTiling.GetTiling(tiling.mmTilingData);

    tiling.set_k0(k0);
    tiling.set_k1(k1);
    tiling.set_k2(k2);
    tiling.set_spatialShape0(spatialShape0);
    tiling.set_spatialShape1(spatialShape1);
    tiling.set_spatialShape2(spatialShape2);
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_coreTaskCount(coreTaskCount);
    tiling.set_bigCoreCount(bigCoreCount);
    tiling.set_singleLoopTask(singleLoopTask);
    tiling.set_totalTaskCount(totalTaskCount);
    tiling.set_availableUBSize(ubSize);
    tiling.set_withKey(withKey);
    tiling.set_gatherBufLen(gatherBufLen);
    tiling.set_scatterBufLen(scatterBufLen);

    return singleLoopTask;
}

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
    
    auto aivNum = ascendplatformInfo.GetCoreNumAiv();
    auto aicNum = ascendplatformInfo.GetCoreNumAic();
    context->SetBlockDim(aicNum);
    context->SetLocalMemorySize(LOCAL_MEM_SIZE);
    ubSize = LOCAL_MEM_SIZE;

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
    int32_t byteSizePerElements = featureDataType == ge::DT_FLOAT16? HALF_BYTE_SIZE : FP32_BYTE_SIZE;

    auto kernelSizeArr = reinterpret_cast<const int64_t*>(kernelSizePtr->GetData());
    auto outSpatialShapeArr = reinterpret_cast<const int64_t*>(outSpatialShapePtr->GetData());
    int32_t k0 = kernelSizeArr[KERNEL_SIZE_IDX_0];
    int32_t k1 = kernelSizeArr[KERNEL_SIZE_IDX_1];
    int32_t k2 = kernelSizeArr[KERNEL_SIZE_IDX_2];
    uint64_t totalTaskCount = featureShapeArr.GetDim(TOTAL_TASK_DIM_IDX);

    matmul_tiling::MatmulApiTiling mmTiling(ascendplatformInfo);
    
    uint32_t singleLoopTask = setTiling(context, tiling, mmTiling, ubSize, totalTaskCount, *inChannelsPtr, *outChannelsPtr, byteSizePerElements,
        aivNum, k0, k1, k2, *batchSizePtr, *withKeyPtr, outSpatialShapeArr[0], outSpatialShapeArr[1], outSpatialShapeArr[2]);

    ADD_TILING_DATA(context, tiling);
    size_t tmpValiedIndicesWorkspaceSize = (k0 * k1 * k2 *  AlignUp(totalTaskCount, 2 * singleLoopTask) * INT32_BYTE_SIZE);
    size_t featureWorkspaceSize = (2 * totalTaskCount * (*outChannelsPtr * FP32_BYTE_SIZE + *inChannelsPtr * FP32_BYTE_SIZE));
    size_t systemWorkspaceSize = ascendplatformInfo.GetLibApiWorkSpaceSize();
    size_t usrWorkSpaceSize = tmpValiedIndicesWorkspaceSize + featureWorkspaceSize;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    CHECK_NULLPTR(currentWorkspace);
    currentWorkspace[0] = systemWorkspaceSize + usrWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
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
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
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
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(SubmSparseConv3dV3);
}