#include "common.h"
#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "sparse_matmul_tiling.h"
using namespace ge;
using namespace matmul_tiling;

namespace {
constexpr float AVALIABLE_UB_RATIO = 0.8;
constexpr int32_t FLOAT_BYTE_SIZE = 4;
constexpr int32_t INT32_BYTE_SIZE = 4;
constexpr int32_t HALF_BYTE_SIZE = 2;
constexpr int32_t BYTE_ALIGN_SIZE = 32;
constexpr float STAGE2_UB_RATIO = 0.2;
constexpr int32_t MAX_MATMUL_TASK_PER_ITER = 256;
};

namespace optiling {

ge::graphStatus TilingForSparseMatmul(gert::TilingContext* context)
{
    SparseMatmulTilingData tiling;
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
    
    auto inputFeaturePtr = context->GetInputTensor(0);
    auto weightPtr = context->GetInputTensor(1);
    auto indicesOffsetPtr = context->GetInputTensor(2);
    auto featureDataTypePtr = context->GetInputDesc(0);
    if (indicesOffsetPtr == nullptr || weightPtr == nullptr || featureDataTypePtr == nullptr || inputFeaturePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    if (aivNum == 0 || aicNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto inputFeatureShape = inputFeaturePtr->GetStorageShape();
    auto weightShape = weightPtr->GetStorageShape();
    auto indicesOffsetShape = indicesOffsetPtr->GetStorageShape();

    auto featureDataType = featureDataTypePtr->GetDataType();
    int32_t byteSizePerElements = featureDataType == ge::DT_FLOAT16?  HALF_BYTE_SIZE : FLOAT_BYTE_SIZE;
    int32_t k0 = weightShape.GetDim(0);
    int32_t k1 = weightShape.GetDim(1);
    int32_t k2 = weightShape.GetDim(2);
    int32_t inChannels = weightShape.GetDim(3);
    int32_t outChannels = weightShape.GetDim(4);
    int32_t kernelSize = k0 * k1 * k2;

    int32_t inChannelsAligned = CeilAlign(inChannels, BYTE_ALIGN_SIZE / byteSizePerElements);
    int32_t outChannelsAligned = CeilAlign(outChannels, BYTE_ALIGN_SIZE / byteSizePerElements);
    int32_t featureChannelsSize = inChannelsAligned > outChannelsAligned? inChannelsAligned : outChannelsAligned;
    int32_t kernelSizeAligned = CeilAlign(kernelSize, BYTE_ALIGN_SIZE / byteSizePerElements);

    int32_t outputTaskCount = indicesOffsetShape.GetDim(0) - 1;
    int32_t outputCoreTaskCount = outputTaskCount / aivNum;
    int32_t outputBigCoreCount = outputTaskCount % aivNum;
    int32_t outputSingleLoopTask = (ubSize - k2 * inChannelsAligned * byteSizePerElements) / ((1 + kernelSizeAligned) * INT32_BYTE_SIZE + inChannelsAligned * byteSizePerElements);
    int32_t featureBufLen = outputSingleLoopTask;
    int32_t matmulTaskPerIter = (outputCoreTaskCount + 1) > MAX_MATMUL_TASK_PER_ITER? MAX_MATMUL_TASK_PER_ITER : (outputCoreTaskCount + 1);
    matmulTaskPerIter = matmulTaskPerIter == 0? 1 : matmulTaskPerIter;

    auto dataType = (byteSizePerElements == FLOAT_BYTE_SIZE) ? matmul_tiling::DataType::DT_FLOAT : matmul_tiling::DataType::DT_FLOAT16;
    matmul_tiling::MatmulApiTiling mm0Tiling(ascendplatformInfo);
    mm0Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    mm0Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    mm0Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    mm0Tiling.SetOrgShape(matmulTaskPerIter, outChannels, inChannels * kernelSize);
    mm0Tiling.SetShape(matmulTaskPerIter, outChannels, inChannels * kernelSize);
    mm0Tiling.SetBias(false);
    mm0Tiling.SetBufferSpace(-1, -1, -1);
    if (mm0Tiling.GetTiling(tiling.mm0TilingData) == -1) {
        return ge::GRAPH_FAILED;
    }
    
    tiling.set_k0(k0);
    tiling.set_k1(k1);
    tiling.set_k2(k2);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);

    tiling.set_outputCoreTaskCount(outputCoreTaskCount);
    tiling.set_outputBigCoreCount(outputBigCoreCount);
    tiling.set_outputSingleLoopTask(outputSingleLoopTask);
    tiling.set_outputTaskCount(outputTaskCount);
    tiling.set_matmulTaskPerIter(matmulTaskPerIter);

    tiling.set_availableUBSize(ubSize);
    tiling.set_aivNum(aivNum);
    tiling.set_featureBufLen(featureBufLen);

    ADD_TILING_DATA(context, tiling);

    size_t systemWorkspaceSize = ascendplatformInfo.GetLibApiWorkSpaceSize();
    size_t usrWorkSpaceSize = static_cast<uint64_t>(aivNum) * matmulTaskPerIter * kernelSize * inChannels * byteSizePerElements;

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    CHECK_NULLPTR(currentWorkspace);
    currentWorkspace[0] = systemWorkspaceSize + usrWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShapeForSparseMatmul(gert::InferShapeContext* context)
{
    auto weightShape = context->GetInputShape(1);
    auto uniqueIndicesOffsetShape = context->GetInputShape(2);
    if (uniqueIndicesOffsetShape == nullptr || weightShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* sparseValueShape = context->GetOutputShape(0);
    gert::Shape* sparseIndicesShape = context->GetOutputShape(1);
    if (sparseValueShape == nullptr || sparseIndicesShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint64_t actualNum = uniqueIndicesOffsetShape->GetDim(0) - 1;
    *sparseValueShape = {actualNum, weightShape->GetDim(3)};
    *sparseIndicesShape = {actualNum, 4};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeForSparseMatmul(gert::InferDataTypeContext* context)
{
    const ge::DataType feature_dtype = context->GetInputDataType(0);
    const ge::DataType indices_dtype = context->GetInputDataType(2);
    context->SetOutputDataType(0, feature_dtype);
    context->SetOutputDataType(1, indices_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class SparseMatmul : public OpDef {
public:
    explicit SparseMatmul(const char* name) : OpDef(name)
    {
        this->Input("features")
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
        this->Input("unique_indices_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("former_sorted_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("sparse_value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("sparse_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForSparseMatmul).SetInferDataType(ge::InferDtypeForSparseMatmul);

        this->AICore().SetTiling(optiling::TilingForSparseMatmul);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SparseMatmul);
}