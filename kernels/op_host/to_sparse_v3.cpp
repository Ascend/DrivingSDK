#include "ge/utils.h"
#include "to_sparse_v3_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace ge;
using namespace matmul_tiling;

namespace optiling {
constexpr uint32_t DTYPE_FP32_BLOCK = 8;
constexpr uint32_t RESERVED_UB_SIZE = 8 * 1024;

ge::graphStatus ToSparseV3Tiling::Init()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, availableUbSize);
    auto weightPtr = tilingContext->GetInputTensor(1);
    auto indicesOffsetPtr = tilingContext->GetInputTensor(2);
    if (indicesOffsetPtr == nullptr || weightPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto weightShape = weightPtr->GetStorageShape();
    auto indicesOffsetShape = indicesOffsetPtr->GetStorageShape();
    uint32_t kernelD = weightShape.GetDim(0);
    uint32_t kernelH = weightShape.GetDim(1);
    uint32_t kernelW = weightShape.GetDim(2);
    kernelIC = weightShape.GetDim(3);
    kernelOC = weightShape.GetDim(4);
    kernelSize = kernelD * kernelH * kernelW;
    actualNum = indicesOffsetShape.GetDim(0) - 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ToSparseV3Tiling::RunKernelTiling()
{
    GetCubeTilingData();
    GetVectorTilingData();
    tilingContext->SetBlockDim((usedVectorCoreNum + 1) / 2);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ToSparseV3Tiling::GetVectorTilingData()
{
    usedVectorCoreNum = aivNum;
    vectorCoreTask = Ceil(actualNum, usedVectorCoreNum);
    usedVectorCoreNum = Ceil(actualNum, vectorCoreTask);
    vectorLastCoreTask = Tail(actualNum, vectorCoreTask);
    uint32_t kernelSizeAlign = AlignUp(kernelSize, DTYPE_FP32_BLOCK);

    moveLen = (availableUbSize - RESERVED_UB_SIZE) / 4 / (kernelSizeAlign + 9 + kernelSize * kernelIC);
    coreRepeatTimes = Ceil(vectorCoreTask, moveLen);
    lastCoreRepeatTimes = Ceil(vectorLastCoreTask, moveLen);
    coreMoveLenTail = Tail(vectorCoreTask, moveLen);
    lastCoreMoveLenTail = Tail(vectorLastCoreTask, moveLen);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ToSparseV3Tiling::GetCubeTilingData()
{
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    MultiCoreMatmulTiling cubeTiling(ascendplatformInfo);

    uint32_t M = actualNum;
    uint32_t N = kernelOC;
    uint32_t K = kernelSize * kernelIC;

    uint32_t originSingleN = N;
    uint32_t originSingleM = Ceil(M, (uint64_t)aivNum);
    uint32_t baseN = originSingleN;
    uint32_t baseM = AlignUp(originSingleM, 16);
    if (baseM > 64) {
        baseM = 64;
    }
    cubeTiling.SetDim(aivNum);
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    cubeTiling.SetOrgShape(M, N, K);
    cubeTiling.SetSingleShape(originSingleM, originSingleN, K);
    cubeTiling.SetShape(originSingleM, originSingleN, K);
    cubeTiling.SetFixSplit(baseM, baseN, -1);
    cubeTiling.SetBufferSpace(-1, -1, -1);
    if (cubeTiling.GetTiling(tilingData.cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ToSparseV3Tiling::SetTilingData()
{
    tilingData.set_usedVectorCoreNum(usedVectorCoreNum);
    tilingData.set_kernelIC(kernelIC);
    tilingData.set_kernelSize(kernelSize);
    tilingData.set_vectorCoreTask(vectorCoreTask);
    tilingData.set_vectorLastCoreTask(vectorLastCoreTask);
    tilingData.set_moveLen(moveLen);
    tilingData.set_coreRepeatTimes(coreRepeatTimes);
    tilingData.set_coreMoveLenTail(coreMoveLenTail);
    tilingData.set_lastCoreRepeatTimes(lastCoreRepeatTimes);
    tilingData.set_lastCoreMoveLenTail(lastCoreMoveLenTail);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    if (tilingContext->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    size_t userWorkspaceSize = actualNum * kernelSize * kernelIC * sizeof(float);
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForToSparseV3(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ToSparseV3Tiling tilingObject(context);
    tilingObject.Init();
    return tilingObject.RunKernelTiling();
}
}

namespace ge {
static ge::graphStatus InferShapeForToSparseV3(gert::InferShapeContext* context)
{
    auto weightShape = context->GetInputShape(1);
    auto indicesOffsetShape = context->GetInputShape(2);
    if (indicesOffsetShape == nullptr || weightShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* sparseValueShape = context->GetOutputShape(0);
    gert::Shape* sparseIndicesShape = context->GetOutputShape(1);
    if (sparseValueShape == nullptr || sparseIndicesShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint64_t actualNum = indicesOffsetShape->GetDim(0) - 1;
    *sparseValueShape = {actualNum, weightShape->GetDim(3)};
    *sparseIndicesShape = {actualNum, 8};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeForToSparseV3(gert::InferDataTypeContext* context)
{
    const ge::DataType feature_dtype = context->GetInputDataType(0);
    const ge::DataType indices_dtype = context->GetInputDataType(2);
    context->SetOutputDataType(0, feature_dtype);
    context->SetOutputDataType(1, indices_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ToSparseV3 : public OpDef {
public:
    explicit ToSparseV3(const char* name) : OpDef(name)
    {
        this->Input("features")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
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
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("sparse_value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("sparse_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForToSparseV3).SetInferDataType(ge::InferDtypeForToSparseV3);

        this->AICore().SetTiling(optiling::TilingForToSparseV3);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(ToSparseV3);
}