#include "ge/utils.h"
#include "sparse_conv3d_grad_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace ge;
using namespace std;
namespace optiling {
constexpr uint64_t DTYPE_FP32_BLOCK = 8;
constexpr uint64_t RESERVED_UB_SIZE = 8 * 1024;

ge::graphStatus SparseConv3dGradV2Tiling::Init()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, availableUbSize);

    auto indicesOffsetPtr = tilingContext->GetInputTensor(0);
    auto featurePtr = tilingContext->GetInputTensor(2);
    auto weightPtr = tilingContext->GetInputTensor(3);
    if (indicesOffsetPtr == nullptr || featurePtr == nullptr || weightPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto featureShape = featurePtr->GetStorageShape();
    auto weightShape = weightPtr->GetStorageShape();
    auto indicesOffsetShape = indicesOffsetPtr->GetStorageShape();

    actualNum = featureShape.GetDim(0);
    uint64_t kernelD = weightShape.GetDim(0);
    uint64_t kernelH = weightShape.GetDim(1);
    uint64_t kernelW = weightShape.GetDim(2);
    kernelIC = weightShape.GetDim(4);
    kernelOC = weightShape.GetDim(3);
    kernelSize = kernelD * kernelH * kernelW;
    vectorActualNum = indicesOffsetShape.GetDim(0) - 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseConv3dGradV2Tiling::RunKernelTiling()
{
    GetCubeTilingData();
    GetVectorTilingData();
    tilingContext->SetBlockDim((usedVectorCoreNum + 1) / 2);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseConv3dGradV2Tiling::MatMulGetData(uint64_t M, uint64_t N, uint64_t K, uint32_t mode)
{
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    MultiCoreMatmulTiling cubeTiling(ascendplatformInfo);
    uint64_t originSingleN = N;
    uint64_t originSingleM = Ceil(M, (uint64_t)aivNum);
    if (originSingleM < 16) {
        originSingleM = 16;
    }
    uint32_t usedCore = Ceil(M, originSingleM);
    uint64_t baseN = originSingleN;
    uint64_t baseM = AlignUp(originSingleM, 16);
    if (baseM > 64) {
        baseM = 64;
    }
    cubeTiling.SetDim(usedCore);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetOrgShape(M, N, K);
    cubeTiling.SetSingleShape(originSingleM, originSingleN, K);
    cubeTiling.SetFixSplit(baseM, baseN, -1);
    cubeTiling.SetBufferSpace(-1, -1, -1);
    if (mode == 0) {
        cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
        if (cubeTiling.GetTiling(tilingData.featureCubeTilingData) == -1) {
            return ge::GRAPH_FAILED;
        }
        featureCubeNum = usedCore;
    }
    if (mode == 1) {
        cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
        if (cubeTiling.GetTiling(tilingData.weightCubeTilingData) == -1) {
            return ge::GRAPH_FAILED;
        }
        weightCubeNum = usedCore;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseConv3dGradV2Tiling::GetCubeTilingData()
{
    int64_t M = actualNum;
    int64_t N = kernelIC;
    int64_t K = kernelSize * kernelOC;
    MatMulGetData(M, N, K, 0);
    M = kernelSize * kernelIC;
    N = kernelOC;
    K = vectorActualNum;
    MatMulGetData(M, N, K, 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseConv3dGradV2Tiling::GetVectorTilingData()
{
    usedVectorCoreNum = aivNum;
    vectorCoreTask = Ceil(vectorActualNum, usedVectorCoreNum);
    usedVectorCoreNum = Ceil(vectorActualNum, vectorCoreTask);
    vectorLastCoreTask = Tail(vectorActualNum, vectorCoreTask);
    uint64_t kernelSizeAlign = AlignUp(kernelSize, DTYPE_FP32_BLOCK);
    uint64_t kernelICAlign = AlignUp(kernelIC, DTYPE_FP32_BLOCK);
    uint64_t featureUb = kernelSize * kernelICAlign * sizeof(float);
    moveLen = (availableUbSize - RESERVED_UB_SIZE - featureUb) / 4 / (kernelSizeAlign + 1);
    coreRepeatTimes = Ceil(vectorCoreTask, moveLen);
    lastCoreRepeatTimes = Ceil(vectorLastCoreTask, moveLen);
    coreMoveLenTail = Tail(vectorCoreTask, moveLen);
    lastCoreMoveLenTail = Tail(vectorLastCoreTask, moveLen);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseConv3dGradV2Tiling::SetTilingData()
{
    tilingData.set_usedVectorCoreNum(usedVectorCoreNum);
    tilingData.set_featureCubeNum(featureCubeNum);
    tilingData.set_weightCubeNum(weightCubeNum);
    tilingData.set_kernelIC(kernelIC);
    tilingData.set_kernelOC(kernelOC);
    tilingData.set_kernelSize(kernelSize);
    tilingData.set_moveLen(moveLen);
    tilingData.set_vectorActualNum(vectorActualNum);
    tilingData.set_vectorCoreTask(vectorCoreTask);
    tilingData.set_vectorLastCoreTask(vectorLastCoreTask);
    tilingData.set_coreRepeatTimes(coreRepeatTimes);
    tilingData.set_coreMoveLenTail(coreMoveLenTail);
    tilingData.set_lastCoreRepeatTimes(lastCoreRepeatTimes);
    tilingData.set_lastCoreMoveLenTail(lastCoreMoveLenTail);
    if (tilingContext->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    size_t workspaceSize = (actualNum * kernelSize * kernelOC + vectorActualNum * kernelSize * kernelIC) * sizeof(float);
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = systemWorkspaceSize + workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForSparseConv3dGradV2(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    SparseConv3dGradV2Tiling tilingObject(context);
    tilingObject.Init();
    return tilingObject.RunKernelTiling();
}
}

namespace ge {
static ge::graphStatus InferShapeForSparseConv3dGradV2(gert::InferShapeContext* context)
{
    const gert::Shape* indiceOffsetShape = context->GetInputShape(0);
    const gert::Shape* featureShape = context->GetInputShape(2);
    const gert::Shape* weightShape = context->GetInputShape(3);
    if (indiceOffsetShape == nullptr || featureShape == nullptr || weightShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint64_t featureNum = featureShape->GetDim(0);
    uint64_t outfeatureNum = indiceOffsetShape->GetDim(0) - 1;
    uint64_t depth = weightShape->GetDim(0);
    uint64_t width = weightShape->GetDim(1);
    uint64_t height = weightShape->GetDim(2);
    uint64_t kernelIC = weightShape->GetDim(4);
    uint64_t kernelOC = weightShape->GetDim(3);
    gert::Shape* featureGradShape = context->GetOutputShape(0);
    gert::Shape* weightGradShape = context->GetOutputShape(1);
    if (featureGradShape == nullptr || weightGradShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *featureGradShape = {featureNum, kernelIC};
    *weightGradShape = {depth * width * height * kernelIC, kernelOC};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeForSparseConv3dGradV2(gert::InferDataTypeContext* context)
{
    const ge::DataType feature_dtype = context->GetInputDataType(2);
    context->SetOutputDataType(0, feature_dtype);
    context->SetOutputDataType(1, feature_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class SparseConv3dGradV2 : public OpDef {
public:
    explicit SparseConv3dGradV2(const char* name) : OpDef(name)
    {
        this->Input("indices_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("former_sorted_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("feature")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("feature_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("weight_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForSparseConv3dGradV2).SetInferDataType(ge::InferDtypeForSparseConv3dGradV2);

        this->AICore().SetTiling(optiling::TilingForSparseConv3dGradV2);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SparseConv3dGradV2);
}