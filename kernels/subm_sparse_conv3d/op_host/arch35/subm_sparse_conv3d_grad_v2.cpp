#include "subm_sparse_conv3d_grad_v2.h"

#include "common/op_host/common.h"
#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace ge;
using namespace std;
using namespace AscendC;


namespace {
const int32_t TOTAL_TASK_DIM_IDX = 0;
const int32_t INT32_BYTE_SIZE = 4;
const int32_t FLOAT32_BYTE_SIZE = 4;
const int32_t FLOAT16_BYTE_SIZE = 2;
const int32_t BYTE_ALIGN_SIZE = 32;
const float AVALIABLE_UB_RATIO = 0.9;
const int32_t SINGLE_LOOP_COMPARE_UB = 256;

const int32_t INPUT_FEATURES_IDX = 0;
const int32_t INPUT_WEIGHT_IDX = 1;
const int32_t INPUT_GRAD_OUT_FEATURES_IDX = 2;
const int32_t INPUT_INDICES_OFFSET_IDX = 3;
const int32_t OUTPUT_FEATURES_GRAD_IDX = 0;
const int32_t OUTPUT_WEIGHT_GRAD_IDX = 1;
const int32_t K0_IDX = 0;
const int32_t K1_IDX = 1;
const int32_t K2_IDX = 2;
const int32_t INCHANNELS_IDX = 3;
const int32_t OUTCHANNELS_IDX = 4;

const int32_t PROCESS_NUM_PER_STEP = 2;
const int32_t BUFFER_NUM = 2;
const int32_t INDICES_MEM = 20 * 1024;
const int32_t REVERSED_MEM = 64 * 1024;
const int32_t REVERSED_USER_WORKSPACE = 16 * 1024 * 1024;
const int32_t AIC_AIV_RATIO = 2;
} // namespace


// define tiling function
namespace optiling {
ge::graphStatus TilingForSubmSparseConv3dGradV2(gert::TilingContext* context)
{
    SubmConv3dGradV2TillingData tilingData;
    CHECK_NULLPTR(context);
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // get vector core number
    auto aivNum = ascendPlatformInfo.GetCoreNumAiv();
    // get cube core number
    auto aicNum = ascendPlatformInfo.GetCoreNumAic();
    if (aivNum == 0 || aicNum == 0) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(aicNum);

    // get shape info
    // features: [N, C1]
    // weight: [K, K, K, C1, C2]
    // gradOutFeatures: [N, C2]
    // indicesOffset: [N * K * K * K]
    const auto featuresShapePtr = context->GetInputShape(INPUT_FEATURES_IDX);
    const auto weightShapePtr = context->GetInputShape(INPUT_WEIGHT_IDX);
    if (featuresShapePtr == nullptr || weightShapePtr == nullptr ||
        context->GetInputShape(INPUT_GRAD_OUT_FEATURES_IDX) == nullptr ||
        context->GetInputShape(INPUT_INDICES_OFFSET_IDX) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto featuresShape = featuresShapePtr->GetStorageShape();
    auto weightShape = weightShapePtr->GetStorageShape();
    int32_t k0 = weightShape.GetDim(K0_IDX);
    int32_t k1 = weightShape.GetDim(K1_IDX);
    int32_t k2 = weightShape.GetDim(K2_IDX);
    int32_t inChannels = weightShape.GetDim(INCHANNELS_IDX);
    int32_t outChannels = weightShape.GetDim(OUTCHANNELS_IDX);

    // get element datatype
    auto featureDataTypePtr = context->GetInputDesc(0);
    auto featureDataType = featureDataTypePtr->GetDataType();
    int32_t byteSizePerElement = featureDataType == ge::DT_FLOAT16 ? FLOAT16_BYTE_SIZE : FLOAT32_BYTE_SIZE;

    // get task count for each vector core
    int64_t totalTaskCount = featuresShape.GetDim(TOTAL_TASK_DIM_IDX);
    int32_t coreTaskCount = totalTaskCount / aicNum;
    int32_t bigCoreCount = totalTaskCount % aicNum;

    int32_t kernelSize = k0 * k1 * k2;
    int32_t halfK = kernelSize / 2;
    int32_t kernelSizeAligned = CeilAlign(kernelSize, BYTE_ALIGN_SIZE / byteSizePerElement);
    int32_t inChannelsAligned = CeilAlign(inChannels, BYTE_ALIGN_SIZE / byteSizePerElement);
    int32_t outChannelsAlinged = CeilAlign(outChannels, BYTE_ALIGN_SIZE / byteSizePerElement);
    int64_t totalTaskAligned = CeilAlign(totalTaskCount, static_cast<int64_t>(BYTE_ALIGN_SIZE / byteSizePerElement));
    int32_t upperCoreTaskCount = coreTaskCount + (bigCoreCount > 0);

    uint64_t ubSize;
    ascendPlatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize = ubSize - REVERSED_MEM;
    context->SetLocalMemorySize(ubSize);
    ubSize *= AVALIABLE_UB_RATIO;

    int32_t usedUBSize = INDICES_MEM;
    const int32_t INT_SPACE_NUM = 5;
    // singleLoopTask表示处理indices的任务量
    int32_t singleLoopTask = (INDICES_MEM - INT_SPACE_NUM * SINGLE_LOOP_COMPARE_UB) / (INT_SPACE_NUM * INT32_BYTE_SIZE);
    singleLoopTask = max(min(singleLoopTask, DivCeil(upperCoreTaskCount, AIC_AIV_RATIO)), 1);
    // innerLoopTask表示处理channels的任务量
    int32_t innerLoopTask =
        (ubSize - usedUBSize) /
        (BUFFER_NUM * PROCESS_NUM_PER_STEP *
            (byteSizePerElement * (inChannelsAligned + outChannelsAlinged) + FLOAT32_BYTE_SIZE * inChannelsAligned));
    if (innerLoopTask <= 0) {
        return ge::GRAPH_FAILED;
    }

    // define matmul tiling
    auto matmul_dtype = byteSizePerElement == FLOAT16_BYTE_SIZE ? matmul_tiling::DataType::DT_FLOAT16 :
                                                                  matmul_tiling::DataType::DT_FLOAT;

    matmul_tiling::MatmulApiTiling featureMatmulTiling(ascendPlatformInfo);
    featureMatmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype);
    featureMatmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype, true);
    featureMatmulTiling.SetCType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    featureMatmulTiling.SetOrgShape(upperCoreTaskCount, inChannels, outChannels);
    featureMatmulTiling.SetShape(upperCoreTaskCount, inChannels, outChannels);
    featureMatmulTiling.SetBias(false);
    featureMatmulTiling.SetBufferSpace(-1, -1, -1);
    if (featureMatmulTiling.GetTiling(tilingData.featureMatmulTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }
    matmul_tiling::MatmulApiTiling weightMatmulTiling(ascendPlatformInfo);

    weightMatmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype, true);
    weightMatmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype);
    weightMatmulTiling.SetCType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    weightMatmulTiling.SetOrgShape(inChannels, outChannels, upperCoreTaskCount);
    weightMatmulTiling.SetShape(inChannels, outChannels, upperCoreTaskCount);
    weightMatmulTiling.SetBias(false);
    weightMatmulTiling.SetBufferSpace(-1, -1, -1);
    if (weightMatmulTiling.GetTiling(tilingData.weightMatmulTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    // set tilingData varialbes
    tilingData.set_k0(k0);
    tilingData.set_k1(k1);
    tilingData.set_k2(k2);
    tilingData.set_inChannels(inChannels);
    tilingData.set_outChannels(outChannels);
    tilingData.set_intSpaceNum(INT_SPACE_NUM);
    tilingData.set_totalTaskCount(totalTaskCount);
    tilingData.set_coreTaskCount(coreTaskCount);
    tilingData.set_bigCoreCount(bigCoreCount);
    tilingData.set_singleLoopTask(singleLoopTask);
    tilingData.set_processNumPerStep(PROCESS_NUM_PER_STEP);
    tilingData.set_innerLoopTask(innerLoopTask);
    tilingData.set_bufferNum(BUFFER_NUM);

    // save to tilingData buffer
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ADD_TILING_DATA(context, tilingData);

    // set workSpaceSize
    size_t systemWorkspaceSize = ascendPlatformInfo.GetLibApiWorkSpaceSize();
    size_t tmpSparseFeaturesWorkSpaceSize = totalTaskCount * inChannels * PROCESS_NUM_PER_STEP * byteSizePerElement;
    size_t tmpFeatureMatmulResWorkSpaceSize = totalTaskCount * inChannels * PROCESS_NUM_PER_STEP * FLOAT32_BYTE_SIZE;
    size_t tmpSparseGradOutFeaturesWorkSpaceSize =
        totalTaskCount * outChannels * PROCESS_NUM_PER_STEP * byteSizePerElement;
    size_t tmpSparseIndicesWorkSpaceSize = halfK * totalTaskCount * INT32_BYTE_SIZE * PROCESS_NUM_PER_STEP;
    size_t tmpSparseNumCountWorkSpaceSize = halfK * totalTaskCount * INT32_BYTE_SIZE;

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    CHECK_NULLPTR(currentWorkspace);
    currentWorkspace[0] = systemWorkspaceSize + tmpFeatureMatmulResWorkSpaceSize + tmpSparseFeaturesWorkSpaceSize +
                          tmpSparseGradOutFeaturesWorkSpaceSize + tmpSparseIndicesWorkSpaceSize +
                          tmpSparseNumCountWorkSpaceSize + REVERSED_USER_WORKSPACE;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling


// define infer shape function
namespace ge {
static ge::graphStatus InferShapeForSubmSparseConv3dGradV2(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* featrueGradShape = context->GetOutputShape(OUTPUT_FEATURES_GRAD_IDX);
    gert::Shape* weightGradShape = context->GetOutputShape(OUTPUT_WEIGHT_GRAD_IDX);
    if (featrueGradShape == nullptr || weightGradShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto featuresShapePtr = context->GetInputShape(INPUT_FEATURES_IDX);
    const auto weightShapePtr = context->GetInputShape(INPUT_WEIGHT_IDX);
    if (featuresShapePtr == nullptr || weightShapePtr == nullptr ||
        context->GetInputShape(INPUT_GRAD_OUT_FEATURES_IDX) == nullptr ||
        context->GetInputShape(INPUT_INDICES_OFFSET_IDX) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int32_t k0 = weightShapePtr->GetDim(K0_IDX);
    int32_t k1 = weightShapePtr->GetDim(K1_IDX);
    int32_t k2 = weightShapePtr->GetDim(K2_IDX);
    int32_t inChannels = weightShapePtr->GetDim(INCHANNELS_IDX);
    int32_t outChannels = weightShapePtr->GetDim(OUTCHANNELS_IDX);
    int32_t totalTaskCount = featuresShapePtr->GetDim(TOTAL_TASK_DIM_IDX);

    // set output dimension
    featrueGradShape->SetDimNum(0);
    featrueGradShape->AppendDim(totalTaskCount);
    featrueGradShape->AppendDim(inChannels);

    weightGradShape->SetDimNum(0);
    weightGradShape->AppendDim(k0);
    weightGradShape->AppendDim(k1);
    weightGradShape->AppendDim(k2);
    weightGradShape->AppendDim(inChannels);
    weightGradShape->AppendDim(outChannels);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForSubmSparseConv3dGradV2(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // upper precision for a5 atomicAdd
    const ge::DataType featuresGradDtype = ge::DT_FLOAT;
    const ge::DataType weightGradDtype = ge::DT_FLOAT;

    context->SetOutputDataType(OUTPUT_FEATURES_GRAD_IDX, featuresGradDtype);
    context->SetOutputDataType(OUTPUT_WEIGHT_GRAD_IDX, weightGradDtype);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge


// op prototype registry
namespace ops {
class SubmSparseConv3dGradV2 : public OpDef {
public:
    explicit SubmSparseConv3dGradV2(const char* name) : OpDef(name)
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
        this->Input("indices_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("features_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("weight_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForSubmSparseConv3dGradV2)
            .SetInferDataType(ge::InferDataTypeForSubmSparseConv3dGradV2);

        this->AICore().SetTiling(optiling::TilingForSubmSparseConv3dGradV2);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(SubmSparseConv3dGradV2);
} // namespace ops
