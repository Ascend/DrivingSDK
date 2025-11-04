#include "subm_sparse_conv3d_grad_v2.h"
#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "common.h"
#include "tiling/platform/platform_ascendc.h"

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
    const float SINGLE_LOOP_UB_SIZE = 7 * 4;
    const int32_t MAX_CV_CYCLE_NUM = 20;
    const int32_t CHANNEL_BUF_LEN = 64;
    const int32_t MIN_SINGLELOOPTASK = 10;
    const int32_t INT_SPACE_NUM = 3;
    const int32_t INCHANNELS_BUF_NUM = 2;

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
}


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
    // indicesOffset: [N*K*K*K]
    const auto featuresShapePtr = context->GetInputShape(INPUT_FEATURES_IDX);
    const auto weightShapePtr = context->GetInputShape(INPUT_WEIGHT_IDX);
    if (featuresShapePtr == nullptr || weightShapePtr == nullptr || 
        context->GetInputShape(INPUT_GRAD_OUT_FEATURES_IDX) == nullptr || context->GetInputShape(INPUT_INDICES_OFFSET_IDX) == nullptr) {
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
    if (featureDataTypePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto featureDataType = featureDataTypePtr->GetDataType();
    int32_t byteSizePerElement = featureDataType == ge::DT_FLOAT16? FLOAT16_BYTE_SIZE : FLOAT32_BYTE_SIZE ;

    // get task count for each vector core
    int32_t totalTaskCount = featuresShape.GetDim(TOTAL_TASK_DIM_IDX);
    int32_t coreTaskCount = totalTaskCount / aivNum;
    int32_t bigCoreCount = totalTaskCount % aivNum;
    
    int32_t kernelSize = k0 * k1 * k2;
    int32_t kernelSizeAligned = CeilAlign(kernelSize, BYTE_ALIGN_SIZE / byteSizePerElement);
    int32_t inChannelsAligned = CeilAlign(inChannels, BYTE_ALIGN_SIZE / byteSizePerElement);
    int32_t outChannelsAlinged = CeilAlign(outChannels, BYTE_ALIGN_SIZE / byteSizePerElement);
    int32_t totalTaskAligned = CeilAlign(totalTaskCount, BYTE_ALIGN_SIZE / byteSizePerElement);
    
    uint64_t ubSize;
    ascendPlatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize *= AVALIABLE_UB_RATIO;

    int32_t singleLoopTask;
    singleLoopTask = (ubSize - INT_SPACE_NUM * SINGLE_LOOP_UB_SIZE) / (INT_SPACE_NUM * INT32_BYTE_SIZE+ 
        (INCHANNELS_BUF_NUM * inChannelsAligned + outChannelsAlinged) * byteSizePerElement);

    // define matmul tiling
    auto matmul_dtype = byteSizePerElement == 2 ? matmul_tiling::DataType::DT_FLOAT16 : matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::MatmulApiTiling featureMatmulTiling(ascendPlatformInfo);
    featureMatmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype);
    featureMatmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype, true);
    featureMatmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype);
    featureMatmulTiling.SetOrgShape(coreTaskCount + (bigCoreCount > 0), inChannels, outChannels);
    featureMatmulTiling.SetShape(coreTaskCount + (bigCoreCount > 0), inChannels, outChannels);
    featureMatmulTiling.SetBias(false);
    featureMatmulTiling.SetBufferSpace(-1, -1, -1);
    if (featureMatmulTiling.GetTiling(tilingData.featureMatmulTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    matmul_tiling::MatmulApiTiling weightMatmulTiling(ascendPlatformInfo);
    weightMatmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype, true);
    weightMatmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype);
    weightMatmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    weightMatmulTiling.SetOrgShape(inChannels, outChannels, coreTaskCount + (bigCoreCount > 0));
    weightMatmulTiling.SetShape(inChannels, outChannels, coreTaskCount + (bigCoreCount > 0));
    weightMatmulTiling.SetBias(false);
    weightMatmulTiling.SetBufferSpace(-1, -1, -1);
    if (weightMatmulTiling.GetTiling(tilingData.weightMatmulTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    // set tilingData varialbes
    tilingData.set_aivNum(aivNum);
    tilingData.set_k0(k0);
    tilingData.set_k1(k1);
    tilingData.set_k2(k2);
    tilingData.set_inChannels(inChannels);
    tilingData.set_outChannels(outChannels);
    tilingData.set_totalTaskCount(totalTaskCount);
    tilingData.set_coreTaskCount(coreTaskCount);
    tilingData.set_bigCoreCount(bigCoreCount);
    tilingData.set_singleLoopTask(singleLoopTask);

    // save to tilingData buffer
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ADD_TILING_DATA(context, tilingData);

    // set workSpaceSize with 2* 
    size_t systemWorkspaceSize = ascendPlatformInfo.GetLibApiWorkSpaceSize();
    size_t tmpSparseFeaturesWorkSpaceSize = totalTaskCount * inChannels * byteSizePerElement;
    size_t tmpFeatureMatmulResWorkSpaceSize = totalTaskCount * inChannels * byteSizePerElement;
    size_t tmpSparseGradOutFeaturesWorkSpaceSize = totalTaskCount * outChannels * byteSizePerElement;
    size_t tmpSparseIndicesWorkSpaceSize = (totalTaskCount + 1) * INT32_BYTE_SIZE;
    
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    CHECK_NULLPTR(currentWorkspace);
    currentWorkspace[0] = systemWorkspaceSize + tmpFeatureMatmulResWorkSpaceSize + 
        tmpSparseFeaturesWorkSpaceSize + tmpSparseGradOutFeaturesWorkSpaceSize + tmpSparseIndicesWorkSpaceSize;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling


// define infer shape function
namespace ge{
static ge::graphStatus InferShapeForSubmSparseConv3dGradV2(gert::InferShapeContext* context) {
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
        context->GetInputShape(INPUT_GRAD_OUT_FEATURES_IDX) == nullptr || context->GetInputShape(INPUT_INDICES_OFFSET_IDX) == nullptr) {
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

static ge::graphStatus InferDataTypeForSubmSparseConv3dGradV2(gert::InferDataTypeContext* context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType featuresGradDtype = context->GetInputDataType(INPUT_FEATURES_IDX);
    const ge::DataType weightGradDtype = ge::DT_FLOAT;

    context->SetOutputDataType(OUTPUT_FEATURES_GRAD_IDX, featuresGradDtype);
    context->SetOutputDataType(OUTPUT_WEIGHT_GRAD_IDX, weightGradDtype);
    return ge::GRAPH_SUCCESS;
}
}


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
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
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
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SubmSparseConv3dGradV2);
} // namespace ops
