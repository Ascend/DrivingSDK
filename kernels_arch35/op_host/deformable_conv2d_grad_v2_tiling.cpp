#include "deformable_conv2d_grad_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "ge/utils.h"

namespace {
constexpr uint8_t INPUT_FEATURES_IDX = 0;
constexpr uint8_t INPUT_WEIGHT_IDX = 1;
constexpr uint8_t INPUT_BIAS_IDX = 2;
constexpr uint8_t INPUT_OFFSET_IDX = 3;
constexpr uint8_t INPUT_MASK_IDX = 4;

constexpr uint8_t OUTPUT_FEATURES_GRAD_IDX = 0;
constexpr uint8_t OUTPUT_WEIGHT_GRAD_IDX = 1;
constexpr uint8_t OUTPUT_BIAS_GRAD_IDX = 2;
constexpr uint8_t OUTPUT_OFFSET_GRAD_IDX = 3;
constexpr uint8_t OUTPUT_MASK_GRAD_IDX = 4;

constexpr uint8_t BATCH_DIM = 0;
constexpr uint8_t H_DIM = 1;
constexpr uint8_t W_DIM = 2;
constexpr uint8_t CHANNELS_DIM = 3;

constexpr uint8_t HALF_BYTE_SIZE = 2;
constexpr uint8_t FLOAT_BYTE_SIZE = 4;

constexpr uint8_t VECTOR_COUNT_PER_GROUP = 2;

constexpr uint8_t L0A_DOUBLE_BUF = 2;
constexpr uint8_t L0B_DOUBLE_BUF = 2;
constexpr uint8_t L0C_DOUBLE_BUF = 2;

constexpr uint8_t A1_DOUBLE_BUF = 2;
constexpr uint8_t B1_DOUBLE_BUF = 2;

constexpr uint8_t WORK_SPACE_BUF_NUM = 2;
} // namespace

using namespace ge;
using namespace std;
using namespace AscendC;

namespace optiling {
ge::graphStatus TilingForDeformableConv2dGradV2(gert::TilingContext* context)
{
    CHECK_NULLPTR(context);
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendPlatformInfo.GetCoreNumAic();
    auto aivNum = ascendPlatformInfo.GetCoreNumAiv();
    if (aicNum == 0 || aivNum == 0) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(aicNum);
    const auto xShapePtr = context->GetInputShape(0);
    const auto offsetShapePtr = context->GetInputShape(3);
    const auto weightShapePtr = context->GetInputShape(1);
    auto inputFeatureDataTypePtr = context->GetInputDesc(INPUT_FEATURES_IDX);

    CHECK_NULLPTR(xShapePtr);
    CHECK_NULLPTR(offsetShapePtr);
    CHECK_NULLPTR(weightShapePtr);
    CHECK_NULLPTR(inputFeatureDataTypePtr);

    auto xShape = xShapePtr->GetStorageShape();
    auto offsetShape = offsetShapePtr->GetStorageShape();
    auto weightShape = weightShapePtr->GetStorageShape();
    auto inputFeatureDataType = inputFeatureDataTypePtr->GetDataType();
    int32_t byteSizePerElements = inputFeatureDataType == ge::DT_FLOAT16? HALF_BYTE_SIZE : FLOAT_BYTE_SIZE;

    int64_t n = xShape.GetDim(0);
    int64_t cIn = xShape.GetDim(3);
    int64_t hIn = xShape.GetDim(1);
    int64_t wIn = xShape.GetDim(2);
    int64_t cOut = weightShape.GetDim(0);
    int64_t hOut = offsetShape.GetDim(1);
    int64_t wOut = offsetShape.GetDim(2);

    auto attrsPtr = context->GetAttrs();
    CHECK_NULLPTR(attrsPtr);
    const auto* kernelSizePtr = attrsPtr->GetListInt(0);
    const auto* stridePtr = attrsPtr->GetListInt(1);
    const auto* paddingPtr = attrsPtr->GetListInt(2);
    const auto* dilationPtr = attrsPtr->GetListInt(3);
    const auto* groupsPtr = attrsPtr->GetInt(4);
    const auto* modulatedPtr = attrsPtr->GetBool(6);

    CHECK_NULLPTR(kernelSizePtr);
    CHECK_NULLPTR(stridePtr);
    CHECK_NULLPTR(paddingPtr);
    CHECK_NULLPTR(dilationPtr);
    CHECK_NULLPTR(modulatedPtr);
    CHECK_NULLPTR(groupsPtr);

    auto kernelSize = kernelSizePtr->GetData();
    auto stride = stridePtr->GetData();
    auto padding = paddingPtr->GetData();
    auto dilation = dilationPtr->GetData();
    int64_t groups = *groupsPtr;
    int64_t kH = kernelSize[0];
    int64_t kW = kernelSize[1];
    int64_t cube0TileTaskCount = 64;
    int64_t cube1TileTaskCount = 64;

    int64_t totalTasks = n * hOut * wOut;
    int64_t singleLoopTask = 16;
    int64_t bigCoreCount = totalTasks % aicNum;
    int64_t coreTaskCount =  totalTasks / aicNum;
    int64_t doubleBuffer = cIn <= 256? 2 : 1;

    context->SetTilingKey(*modulatedPtr);
    auto featureDataType = byteSizePerElements == FLOAT_BYTE_SIZE? matmul_tiling::DataType::DT_FLOAT : matmul_tiling::DataType::DT_FLOAT16;
    DeformableConv2dGradV2TilingData tilingData;

    matmul_tiling::MatmulApiTiling mm0Tiling(ascendPlatformInfo);
    matmul_tiling::MatmulApiTiling mm1Tiling(ascendPlatformInfo);
    mm0Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm0Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm0Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm0Tiling.SetOrgShape(cube0TileTaskCount * VECTOR_COUNT_PER_GROUP, kH * kW * cIn, cOut);
    mm0Tiling.SetShape(cube0TileTaskCount * VECTOR_COUNT_PER_GROUP, kH * kW * cIn, cOut);
    mm0Tiling.SetBias(false);
    mm0Tiling.SetBufferSpace(-1, -1, -1);
    if (mm0Tiling.GetTiling(tilingData.mm0TilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    int32_t baseM0 = 256;
    int32_t baseN0 = 256;
    int32_t baseK0 = 32;
    int32_t stepM0 = 1;
    int32_t stepN0 = 1;
    int32_t stepKa0 = 4;
    int32_t stepKb0 = 4;

    tilingData.mm0TilingData.set_dbL0A(L0A_DOUBLE_BUF);
    tilingData.mm0TilingData.set_dbL0B(L0B_DOUBLE_BUF);
    tilingData.mm0TilingData.set_dbL0C(L0C_DOUBLE_BUF);
    tilingData.mm0TilingData.set_baseM(baseM0);
    tilingData.mm0TilingData.set_baseN(baseN0);
    tilingData.mm0TilingData.set_baseK(baseK0);
    tilingData.mm0TilingData.set_stepM(stepM0);
    tilingData.mm0TilingData.set_stepN(stepN0);
    tilingData.mm0TilingData.set_stepKa(stepKa0);
    tilingData.mm0TilingData.set_stepKb(stepKb0);
    tilingData.mm0TilingData.set_depthA1(A1_DOUBLE_BUF * stepM0 * stepKa0);
    tilingData.mm0TilingData.set_depthB1(B1_DOUBLE_BUF * stepN0 * stepKb0);

    mm1Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType, true);
    mm1Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm1Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm1Tiling.SetOrgShape(cOut, kH * kW * cIn, cube1TileTaskCount * VECTOR_COUNT_PER_GROUP);
    mm1Tiling.SetShape(cOut, kH * kW * cIn, cube1TileTaskCount * VECTOR_COUNT_PER_GROUP);
    mm1Tiling.SetBias(false);
    mm1Tiling.SetBufferSpace(-1, -1, -1);
    if (mm1Tiling.GetTiling(tilingData.mm1TilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    int32_t baseM1 = 256;
    int32_t baseN1 = 256;
    int32_t baseK1 = 32;
    int32_t stepM1 = 1;
    int32_t stepN1 = 1;
    int32_t stepKa1 = (cube1TileTaskCount * VECTOR_COUNT_PER_GROUP) / baseK1;
    int32_t stepKb1 = (cube1TileTaskCount * VECTOR_COUNT_PER_GROUP) / baseK1;
    int32_t depthA1 = A1_DOUBLE_BUF * stepM1 * stepKa1;
    int32_t depthB1 = B1_DOUBLE_BUF * stepN1 * stepKb1;

    tilingData.mm1TilingData.set_baseM(baseM1);
    tilingData.mm1TilingData.set_baseN(baseN1);
    tilingData.mm1TilingData.set_baseK(baseK1);
    tilingData.mm1TilingData.set_stepM(stepM1);
    tilingData.mm1TilingData.set_stepN(stepM1);
    tilingData.mm1TilingData.set_stepKa(stepKa1);
    tilingData.mm1TilingData.set_stepKb(stepKb1);
    tilingData.mm1TilingData.set_depthA1(depthA1);
    tilingData.mm1TilingData.set_depthB1(depthB1);

    tilingData.set_n(n);
    tilingData.set_cIn(cIn);
    tilingData.set_hIn(hIn);
    tilingData.set_wIn(wIn);
    tilingData.set_cOut(cOut);
    tilingData.set_hOut(hOut);
    tilingData.set_wOut(wOut);
    tilingData.set_kH(kH);
    tilingData.set_kW(kW);
    tilingData.set_padH(padding[0]);
    tilingData.set_padW(padding[1]);
    tilingData.set_strideH(stride[0]);
    tilingData.set_strideW(stride[1]);
    tilingData.set_dilationH(dilation[0]);
    tilingData.set_dilationW(dilation[1]);
    tilingData.set_groups(groups);
    tilingData.set_doubleBuffer(doubleBuffer);

    tilingData.set_coreCount(aivNum);
    tilingData.set_cube1TileTaskCount(cube1TileTaskCount);
    tilingData.set_cube0TileTaskCount(cube0TileTaskCount);
    tilingData.set_singleLoopTask(singleLoopTask);
    tilingData.set_bigCoreCount(bigCoreCount);
    tilingData.set_coreTaskCount(coreTaskCount);
    ADD_TILING_DATA(context, tilingData);

    size_t systemWorkspaceSize = ascendPlatformInfo.GetLibApiWorkSpaceSize();
    size_t usrWorkSpaceSize = n * hOut * wOut * kH * kW * cIn * WORK_SPACE_BUF_NUM * byteSizePerElements;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    CHECK_NULLPTR(currentWorkspace);
    currentWorkspace[0] = systemWorkspaceSize + usrWorkSpaceSize;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape4DeformableConv2dGradV2(gert::InferShapeContext* context)
{
    CHECK_NULLPTR(context);
    const gert::Shape* xShape = context->GetInputShape(INPUT_FEATURES_IDX);
    const gert::Shape* weightShape = context->GetInputShape(INPUT_WEIGHT_IDX);
    const gert::Shape* offsetShape = context->GetInputShape(INPUT_OFFSET_IDX);
    const gert::Shape* maskShape = context->GetInputShape(INPUT_MASK_IDX);
    CHECK_NULLPTR(xShape);
    CHECK_NULLPTR(weightShape);
    CHECK_NULLPTR(offsetShape);
    CHECK_NULLPTR(maskShape);

    int64_t batchSize = xShape->GetDim(BATCH_DIM);
    int64_t hIn = xShape->GetDim(H_DIM);
    int64_t wIn = xShape->GetDim(W_DIM);
    int64_t inChannels = xShape->GetDim(CHANNELS_DIM);

    gert::Shape* xGradShape = context->GetOutputShape(OUTPUT_FEATURES_GRAD_IDX);
    gert::Shape* weightGradShape = context->GetOutputShape(OUTPUT_WEIGHT_GRAD_IDX);
    gert::Shape* offsetGradShape = context->GetOutputShape(OUTPUT_OFFSET_GRAD_IDX);
    gert::Shape* maskGradShape = context->GetOutputShape(OUTPUT_MASK_GRAD_IDX);
    CHECK_NULLPTR(xGradShape);
    CHECK_NULLPTR(weightGradShape);
    CHECK_NULLPTR(offsetGradShape);
    CHECK_NULLPTR(maskGradShape);

    *xGradShape = *xShape;
    *weightGradShape = *weightShape;
    *offsetGradShape = *offsetShape;
    *maskGradShape = *maskShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4DeformableConv2dGradV2(gert::InferDataTypeContext *context)
{
    const ge::DataType featureDtype = context->GetInputDataType(INPUT_FEATURES_IDX);
    context->SetOutputDataType(OUTPUT_FEATURES_GRAD_IDX, featureDtype);
    context->SetOutputDataType(OUTPUT_WEIGHT_GRAD_IDX, featureDtype);
    context->SetOutputDataType(OUTPUT_OFFSET_GRAD_IDX, featureDtype);
    context->SetOutputDataType(OUTPUT_MASK_GRAD_IDX, featureDtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class DeformableConv2dGradV2 : public OpDef {
public:
    explicit DeformableConv2dGradV2(const char* name) : OpDef(name)
    {
        this->Input("inputFeatures")
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
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gradOutFeatures")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("kernel_size").ListInt();
        this->Attr("stride").ListInt();
        this->Attr("padding").ListInt();
        this->Attr("dilation").ListInt();
        this->Attr("groups").Int();
        this->Attr("deformable_groups").Int();
        this->Attr("modulated").Bool();
        this->Attr("with_bias").Bool();

        this->Output("gradInputFeatures")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("gradWeight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("gradBias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("gradOffset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("gradMask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        
        this->SetInferShape(ge::InferShape4DeformableConv2dGradV2)
            .SetInferDataType(ge::InferDataType4DeformableConv2dGradV2);
        this->AICore().SetTiling(optiling::TilingForDeformableConv2dGradV2);
        this->AICore().AddConfig("ascend910_95");
    }
};

OP_ADD(DeformableConv2dGradV2);
} // namespace ops
