#include "common.h"
#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "sparse_conv3d_grad_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace ge;
using namespace std;
using namespace AscendC;


namespace {
const uint32_t BYTE_ALIGN_SIZE = 32;
const int32_t INT32_BYTE_SIZE = 4;
constexpr uint32_t MINI_TASK_BLOCK = 16;
constexpr uint64_t RESERVED_UB_SIZE = 16 * 1024;

constexpr uint8_t VECTOR_CUBE_RATIO = 2;
constexpr uint32_t USED_POINTS_WORKSPACE = 3;
constexpr uint32_t KERNEL_WORKSPACE = 2;
} // namespace


// define tiling function
namespace optiling {
ge::graphStatus TilingForSparseConv3dGrad(gert::TilingContext* context)
{
    CHECK_NULLPTR(context);
    SparseConv3dGradTillingData tilingData;
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aivNum = ascendPlatformInfo.GetCoreNumAiv();
    auto aicNum = ascendPlatformInfo.GetCoreNumAic();
    if (aivNum == 0 || aicNum == 0) {
        return ge::GRAPH_FAILED;
    }

    const auto featuresShapePtr = context->GetInputShape(0);
    const auto weightShapePtr = context->GetInputShape(1);
    const auto indicesOffsetShapePtr = context->GetInputShape(4);
    CHECK_NULLPTR(featuresShapePtr);
    CHECK_NULLPTR(weightShapePtr);
    CHECK_NULLPTR(indicesOffsetShapePtr);
    auto featuresShape = featuresShapePtr->GetStorageShape();
    auto weightShape = weightShapePtr->GetStorageShape();
    auto indicesOffsetShape = indicesOffsetShapePtr->GetStorageShape();

    uint32_t inputPointsNum = featuresShape.GetDim(0);
    uint32_t outPointsNum = indicesOffsetShape.GetDim(0) - 1;
    if (inputPointsNum == 0 || outPointsNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t k0 = weightShape.GetDim(0);
    uint32_t k1 = weightShape.GetDim(1);
    uint32_t k2 = weightShape.GetDim(2);
    uint32_t inChannels = weightShape.GetDim(3);
    uint32_t outChannels = weightShape.GetDim(4);
    uint32_t kernelSize = k0 * k1 * k2;

    uint32_t kernelSizeAlign32 = CeilAlign(kernelSize, static_cast<uint32_t>(32));
    uint32_t tmpSortSize = GetSortTmpSize(ascendPlatformInfo, kernelSizeAlign32, 4);
    if (tmpSortSize == 0) {
        return ge::GRAPH_FAILED;
    }
    tilingData.set_kernelSizeAlign32(kernelSizeAlign32);
    tilingData.set_tmpSortSize(tmpSortSize);

    uint64_t featuresGradSize = inputPointsNum * inChannels;
    uint64_t weightGradSize = kernelSize * outPointsNum * inputPointsNum;
    tilingData.set_featuresGradSize(featuresGradSize);
    tilingData.set_weightGradSize(weightGradSize);

    // get element datatype
    auto featureDataTypePtr = context->GetInputDesc(0);
    CHECK_NULLPTR(featureDataTypePtr);
    auto featureDataType = featureDataTypePtr->GetDataType();
    uint32_t byteSizePerElement = (featureDataType == ge::DT_FLOAT16) ? 2 : 4;

    uint64_t availableUbSize;
    ascendPlatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, availableUbSize);
    availableUbSize = availableUbSize - RESERVED_UB_SIZE;
    uint32_t sparseRatio = 1;
    uint32_t ubMaxTaskNum = availableUbSize / ((2 * inChannels + outChannels) * byteSizePerElement +
                                                  (4 + 2 * kernelSize) * sparseRatio * INT32_BYTE_SIZE);
    ubMaxTaskNum = ubMaxTaskNum < MINI_TASK_BLOCK ? ubMaxTaskNum : FloorAlign(ubMaxTaskNum, MINI_TASK_BLOCK);
    if (ubMaxTaskNum == 0) {
        return ge::GRAPH_FAILED;
    }

    // core segment
    uint32_t mainCoreTask = Ceil(outPointsNum, aivNum);
    mainCoreTask = CeilAlign(mainCoreTask, ubMaxTaskNum);
    uint32_t usedVectorNum = Ceil(outPointsNum, mainCoreTask);
    uint32_t lastCoreTask = Tail(outPointsNum, mainCoreTask);
    if (lastCoreTask == 0) {
        lastCoreTask = mainCoreTask;
    }

    uint32_t totalTaskNum = outPointsNum;
    uint64_t gradOutWorkSpaceOffset = 0;
    uint64_t featuresWorkSpaceOffset = totalTaskNum * outChannels;
    uint64_t tmpGradFeaturesWorkSpaceOffset = featuresWorkSpaceOffset + totalTaskNum * inChannels;

    uint64_t startIndicesWorkSpaceOffset =
        tmpGradFeaturesWorkSpaceOffset + totalTaskNum * inChannels; // recore the start
    uint64_t endIndicesWorkSpaceOffset = startIndicesWorkSpaceOffset + totalTaskNum;
    uint64_t inputIndicesPtrWorkSpaceOffset = endIndicesWorkSpaceOffset + totalTaskNum;

    uint64_t inputIndicesWorkSpaceOffset = inputIndicesPtrWorkSpaceOffset + totalTaskNum;
    uint64_t kernelIndicesWorkSpaceOffset = inputIndicesWorkSpaceOffset + totalTaskNum * kernelSize;

    tilingData.set_featuresWorkSpaceOffset(featuresWorkSpaceOffset);
    tilingData.set_tmpGradFeaturesWorkSpaceOffset(tmpGradFeaturesWorkSpaceOffset);
    tilingData.set_startIndicesWorkSpaceOffset(startIndicesWorkSpaceOffset);
    tilingData.set_endIndicesWorkSpaceOffset(endIndicesWorkSpaceOffset);
    tilingData.set_inputIndicesPtrWorkSpaceOffset(inputIndicesPtrWorkSpaceOffset);
    tilingData.set_inputIndicesWorkSpaceOffset(inputIndicesWorkSpaceOffset);
    tilingData.set_kernelIndicesWorkSpaceOffset(kernelIndicesWorkSpaceOffset);

    // define matmul tiling
    auto matmul_dtype =
        (byteSizePerElement == 2) ? matmul_tiling::DataType::DT_FLOAT16 : matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::MatmulApiTiling featureMatmulTiling(ascendPlatformInfo);
    featureMatmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype);
    featureMatmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype, true);
    featureMatmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype);
    featureMatmulTiling.SetOrgShape(mainCoreTask, inChannels, outChannels);
    featureMatmulTiling.SetShape(mainCoreTask, inChannels, outChannels);
    featureMatmulTiling.SetBias(false);
    featureMatmulTiling.SetBufferSpace(-1, -1, -1);
    if (featureMatmulTiling.GetTiling(tilingData.featureMatmulTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    matmul_tiling::MatmulApiTiling weightMatmulTiling(ascendPlatformInfo);
    weightMatmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype, true);
    weightMatmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype);
    weightMatmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype);
    weightMatmulTiling.SetOrgShape(inChannels, outChannels, mainCoreTask);
    weightMatmulTiling.SetShape(inChannels, outChannels, mainCoreTask);
    weightMatmulTiling.SetBias(false);
    weightMatmulTiling.SetBufferSpace(-1, -1, -1);
    if (weightMatmulTiling.GetTiling(tilingData.weightMatmulTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim((usedVectorNum + 1) / VECTOR_CUBE_RATIO);
    tilingData.set_usedVectorNum(usedVectorNum);
    tilingData.set_kernelSize(kernelSize);
    tilingData.set_totalTaskNum(totalTaskNum);
    tilingData.set_inChannels(inChannels);
    tilingData.set_outChannels(outChannels);
    tilingData.set_sparseRatio(sparseRatio);
    tilingData.set_ubMaxTaskNum(ubMaxTaskNum);
    tilingData.set_mainCoreTask(mainCoreTask);
    tilingData.set_lastCoreTask(lastCoreTask);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ADD_TILING_DATA(context, tilingData);

    size_t systemWorkspaceSize = static_cast<size_t>(ascendPlatformInfo.GetLibApiWorkSpaceSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    CHECK_NULLPTR(currentWorkspace);
    currentWorkspace[0] =
        systemWorkspaceSize + static_cast<size_t>(startIndicesWorkSpaceOffset) * byteSizePerElement +
        static_cast<size_t>(totalTaskNum) * (USED_POINTS_WORKSPACE + KERNEL_WORKSPACE * kernelSize) * INT32_BYTE_SIZE;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling


// define infer shape function
namespace ge {
static ge::graphStatus InferShapeForSparseConv3dGrad(gert::InferShapeContext* context)
{
    const gert::Shape* featuresShapePtr = context->GetInputShape(0);
    const gert::Shape* weightShapePtr = context->GetInputShape(1);
    if (featuresShapePtr == nullptr || weightShapePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t inputPointsNum = featuresShapePtr->GetDim(0);
    uint32_t k0 = weightShapePtr->GetDim(0);
    uint32_t k1 = weightShapePtr->GetDim(1);
    uint32_t k2 = weightShapePtr->GetDim(2);
    uint32_t inChannels = weightShapePtr->GetDim(3);
    uint32_t outChannels = weightShapePtr->GetDim(4);

    gert::Shape* featuresGradShape = context->GetOutputShape(0);
    gert::Shape* weightGradShape = context->GetOutputShape(1);
    if (featuresGradShape == nullptr || weightGradShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    *featuresGradShape = {inputPointsNum, inChannels};
    *weightGradShape = {k0 * k1 * k2 * inChannels, outChannels};

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForSparseConv3dGrad(gert::InferDataTypeContext* context)
{
    CHECK_NULLPTR(context)
    const ge::DataType features_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, features_dtype);
    context->SetOutputDataType(1, features_dtype);
    return GRAPH_SUCCESS;
}
} // namespace ge


// op prototype registry
namespace ops {
class SparseConv3dGrad : public OpDef {
public:
    explicit SparseConv3dGrad(const char* name) : OpDef(name)
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
        this->Input("grad_out_features")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("former_sorted_indices")
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

        this->Output("features_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("weight_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForSparseConv3dGrad).SetInferDataType(ge::InferDataTypeForSparseConv3dGrad);
        this->AICore().SetTiling(optiling::TilingForSparseConv3dGrad);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SparseConv3dGrad);
} // namespace ops
