#include "sparse_conv3d_grad_tiling.h"

#include "common/op_host/common.h"
#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace ge;
using namespace std;
using namespace AscendC;


namespace {
const uint8_t INT32_SIZE = 4;
constexpr uint32_t MINI_TASK_BLOCK = 16;
constexpr uint64_t DCACHE_UB_SIZE = 40 * 1024;
constexpr uint64_t RESERVED_UB_SIZE = 16 * 1024;
constexpr uint32_t CONSTANT_K = 1024;

constexpr uint8_t VECTOR_CUBE_RATIO = 2;
constexpr uint32_t WORKSPACE_BUFF = 2;
constexpr uint32_t INDICES_BUFFER = 3;
} // namespace


// define tiling function
namespace optiling {
ge::graphStatus TilingForSparseConv3dGrad(gert::TilingContext* context)
{
    CHECK_NULLPTR(context);
    SparseConv3dGradTilingData tilingData;
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aivNum = ascendPlatformInfo.GetCoreNumAiv();
    auto aicNum = ascendPlatformInfo.GetCoreNumAic();
    if (aivNum == 0 || aicNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto attrsPtr = context->GetAttrs();
    CHECK_NULLPTR(attrsPtr);
    const auto* startOffsetPtr = attrsPtr->GetInt(0);
    const auto* endOffsetPtr = attrsPtr->GetInt(1);
    CHECK_NULLPTR(startOffsetPtr);
    CHECK_NULLPTR(endOffsetPtr);
    int32_t startOffset = *startOffsetPtr;
    int32_t endOffset = *endOffsetPtr;
    int32_t totalPointsCount = endOffset - startOffset;

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

    // get element datatype
    auto featureDataTypePtr = context->GetInputDesc(0);
    CHECK_NULLPTR(featureDataTypePtr);
    auto featureDataType = featureDataTypePtr->GetDataType();
    uint32_t featureByteSize = (featureDataType == ge::DT_FLOAT16) ? 2 : 4;

    uint64_t ubSize;
    ascendPlatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t availableUbSize = ubSize - DCACHE_UB_SIZE;
    context->SetLocalMemorySize(availableUbSize);
    uint32_t loopPointCount = 2048;
    availableUbSize = availableUbSize - RESERVED_UB_SIZE - INDICES_BUFFER * loopPointCount * INT32_SIZE;
    uint64_t oneTaskSize = (2 * inChannels + outChannels) * featureByteSize;
    uint32_t ubMaxTaskNum = availableUbSize / oneTaskSize;
    ubMaxTaskNum = ubMaxTaskNum < MINI_TASK_BLOCK ? ubMaxTaskNum : FloorAlign(ubMaxTaskNum, MINI_TASK_BLOCK);
    if (ubMaxTaskNum == 0) {
        return ge::GRAPH_FAILED;
    }

    // core segment
    uint32_t usedVectorNum = min((int32_t)(aivNum), Ceil(totalPointsCount, loopPointCount));
    uint32_t usedCubeNum = (usedVectorNum + 1) / VECTOR_CUBE_RATIO;
    usedVectorNum = VECTOR_CUBE_RATIO * usedCubeNum;
    uint64_t featureWspOffset = loopPointCount * (2 * inChannels + outChannels);
    uint32_t sparseWspOffset = AlignUp(usedVectorNum * kernelSize, 16);

    // define matmul tiling
    auto matmul_dtype =
        (featureByteSize == 2) ? matmul_tiling::DataType::DT_FLOAT16 : matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::MatmulApiTiling featureMatmulTiling(ascendPlatformInfo);
    featureMatmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype);
    featureMatmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype, true);
    featureMatmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype);
    featureMatmulTiling.SetOrgShape(ubMaxTaskNum, inChannels, outChannels);
    featureMatmulTiling.SetShape(ubMaxTaskNum, inChannels, outChannels);
    featureMatmulTiling.SetBias(false);
    featureMatmulTiling.SetBufferSpace(-1, -1, -1);
    if (featureMatmulTiling.GetTiling(tilingData.featureMatmulTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    matmul_tiling::MatmulApiTiling weightMatmulTiling(ascendPlatformInfo);
    weightMatmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype, true);
    weightMatmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype);
    weightMatmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype);
    weightMatmulTiling.SetOrgShape(inChannels, outChannels, CONSTANT_K);
    weightMatmulTiling.SetShape(inChannels, outChannels, CONSTANT_K);
    weightMatmulTiling.SetBias(false);
    weightMatmulTiling.SetBufferSpace(-1, -1, -1);
    if (weightMatmulTiling.GetTiling(tilingData.weightMatmulTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(usedCubeNum);
    tilingData.set_usedVectorNum(usedVectorNum);
    tilingData.set_kernelSize(kernelSize);
    tilingData.set_totalTaskNum(outPointsNum);
    tilingData.set_totalPointsCount(totalPointsCount);
    tilingData.set_startOffset(startOffset);
    tilingData.set_inChannels(inChannels);
    tilingData.set_outChannels(outChannels);
    tilingData.set_ubMaxTaskNum(ubMaxTaskNum);
    tilingData.set_loopPointCount(loopPointCount);
    tilingData.set_featureWspOffset(featureWspOffset);
    tilingData.set_sparseWspOffset(sparseWspOffset);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ADD_TILING_DATA(context, tilingData);

    size_t systemWorkspaceSize = static_cast<size_t>(ascendPlatformInfo.GetLibApiWorkSpaceSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    CHECK_NULLPTR(currentWorkspace);
    currentWorkspace[0] = systemWorkspaceSize +
                          static_cast<size_t>(totalPointsCount) * (WORKSPACE_BUFF * INT32_SIZE + 1) +
                          static_cast<size_t>(sparseWspOffset) * INT32_SIZE +
                          static_cast<size_t>(featureWspOffset) * usedVectorNum * featureByteSize;
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

        this->Attr("start_offset").AttrType(REQUIRED).Int();
        this->Attr("end_offset").AttrType(REQUIRED).Int();

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
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(SparseConv3dGrad);
} // namespace ops
