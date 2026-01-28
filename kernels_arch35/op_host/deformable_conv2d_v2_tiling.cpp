#include "deformable_conv2d_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "ge/utils.h"

namespace {
constexpr uint8_t INPUT_FEATURES_INDEX = 0;
constexpr uint8_t INPUT_OFFSET_INDEX = 1;
constexpr uint8_t INPUT_MASK_INDEX = 2;
constexpr uint8_t INPUT_WEIGHT_INDEX = 3;
constexpr uint8_t OUTPUT_Y_INDEX = 0;
constexpr uint8_t OUTPUT_OFFSET_INDEX = 1;

constexpr uint8_t DIM_ZERO = 0;
constexpr uint8_t DIM_ONE = 1;
constexpr uint8_t DIM_TWO = 2;
constexpr uint8_t DIM_THREE = 3;

constexpr uint8_t ATTR_KERNEL_DIM = 0;
constexpr uint8_t ATTR_STRIDE_DIM = 1;
constexpr uint8_t ATTR_PADDING_DIM = 2;
constexpr uint8_t ATTR_DILATION_DIM = 3;
constexpr uint8_t ATTR_GROUPS_DIM = 4;
constexpr uint8_t ATTR_MODULATED_DIM = 6;

constexpr uint8_t FLOAT_BYTE_SIZE = 4;
constexpr uint8_t HALF_BYTE_SIZE = 2;

constexpr uint8_t VECTOR_COUNT_PER_GROUP = 2;

constexpr uint8_t L0A_DOUBLE_BUF = 2;
constexpr uint8_t L0B_DOUBLE_BUF = 2;
constexpr uint8_t L0C_DOUBLE_BUF = 2;

constexpr uint8_t STEPKA = 2;
constexpr uint8_t STEPKB= 2;
constexpr uint8_t DEPTHA1 = 4;
constexpr uint8_t DEPTHB1 = 4;
} // namespace

using namespace ge;
using namespace std;
using namespace AscendC;

namespace optiling {
ge::graphStatus TilingForDeformableConv2dV2(gert::TilingContext* context)
{
    CHECK_NULLPTR(context);
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    
    auto aicNum = ascendPlatformInfo.GetCoreNumAic();
    auto aivNum = ascendPlatformInfo.GetCoreNumAiv();
    if (aicNum == 0 || aivNum == 0) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(aicNum);
    const auto xShapePtr = context->GetInputShape(INPUT_FEATURES_INDEX);
    const auto offsetShapePtr = context->GetInputShape(INPUT_OFFSET_INDEX);
    const auto weightShapePtr = context->GetInputShape(INPUT_WEIGHT_INDEX);
    auto inputFeatureDataTypePtr = context->GetInputDesc(INPUT_FEATURES_INDEX);

    CHECK_NULLPTR(xShapePtr);
    CHECK_NULLPTR(offsetShapePtr);
    CHECK_NULLPTR(weightShapePtr);
    CHECK_NULLPTR(inputFeatureDataTypePtr);

    auto xShape = xShapePtr->GetStorageShape();
    auto offsetShape = offsetShapePtr->GetStorageShape();
    auto weightShape = weightShapePtr->GetStorageShape();
    auto inputFeatureDataType = inputFeatureDataTypePtr->GetDataType();
    int64_t byteSizePerElements = inputFeatureDataType == ge::DT_FLOAT16? HALF_BYTE_SIZE : FLOAT_BYTE_SIZE;

    int64_t n = xShape.GetDim(DIM_ZERO);
    int64_t hIn = xShape.GetDim(DIM_ONE);
    int64_t wIn = xShape.GetDim(DIM_TWO);
    int64_t cIn = xShape.GetDim(DIM_THREE);
    int64_t hOut = offsetShape.GetDim(DIM_ONE);
    int64_t wOut = offsetShape.GetDim(DIM_TWO);
    int64_t cOut = weightShape.GetDim(DIM_ZERO);

    auto attrsPtr = context->GetAttrs();
    CHECK_NULLPTR(attrsPtr);
    const auto* kernelSizePtr = attrsPtr->GetListInt(ATTR_KERNEL_DIM);
    const auto* stridePtr = attrsPtr->GetListInt(ATTR_STRIDE_DIM);
    const auto* paddingPtr = attrsPtr->GetListInt(ATTR_PADDING_DIM);
    const auto* dilationPtr = attrsPtr->GetListInt(ATTR_DILATION_DIM);
    const auto* groupsPtr = attrsPtr->GetInt(ATTR_GROUPS_DIM);
    const auto* modulatedPtr = attrsPtr->GetBool(ATTR_MODULATED_DIM);

    CHECK_NULLPTR(kernelSizePtr)
    CHECK_NULLPTR(stridePtr)
    CHECK_NULLPTR(paddingPtr)
    CHECK_NULLPTR(dilationPtr)
    CHECK_NULLPTR(modulatedPtr)
    CHECK_NULLPTR(groupsPtr)

    auto kernelSize = kernelSizePtr->GetData();
    auto stride = stridePtr->GetData();
    auto padding = paddingPtr->GetData();
    auto dilation = dilationPtr->GetData();
    int64_t groups = *groupsPtr;
    int64_t kH = kernelSize[0];
    int64_t kW = kernelSize[1];

    // tiling
    int64_t totalTasks = n * hOut * wOut;
    int64_t singleLoopTask = 32;
    int64_t coreTaskCount = totalTasks / aicNum;
    int64_t bigCoreCount = totalTasks % aicNum;
    int64_t cubeTileTaskCount = AlignUp(coreTaskCount / 2 / 16 + 1, 32 * cIn / 256);

    context->SetTilingKey(*modulatedPtr);
    auto featureDataType = byteSizePerElements == FLOAT_BYTE_SIZE? matmul_tiling::DataType::DT_FLOAT : matmul_tiling::DataType::DT_FLOAT16;
    DeformableConv2dV2TilingData tilingData;
    
    matmul_tiling::MatmulApiTiling mmTiling(ascendPlatformInfo);
    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mmTiling.SetOrgShape(cubeTileTaskCount * VECTOR_COUNT_PER_GROUP, cOut, kH * kW * cIn);
    mmTiling.SetShape(cubeTileTaskCount * VECTOR_COUNT_PER_GROUP, cOut, kH * kW * cIn);
    mmTiling.SetBias(false);
    mmTiling.SetBufferSpace(-1, -1, -1);
    if (mmTiling.GetTiling(tilingData.mmTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    int64_t L0Size = 64 * 1024 / byteSizePerElements;
    int64_t baseM = min(static_cast<int64_t>(256), AlignUp(cubeTileTaskCount * 2, static_cast<int64_t>(128)));
    int64_t baseN = min(static_cast<int64_t>(256), AlignUp(cOut, static_cast<int64_t>(128)));

    if (baseM == 0 || baseN == 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t baseK = min(L0Size / baseM / 2, L0Size / baseN / 2);

    tilingData.mmTilingData.set_dbL0A(L0A_DOUBLE_BUF);
    tilingData.mmTilingData.set_dbL0B(L0B_DOUBLE_BUF);
    tilingData.mmTilingData.set_dbL0C(L0C_DOUBLE_BUF);
    tilingData.mmTilingData.set_baseM(baseM);
    tilingData.mmTilingData.set_baseN(baseN);
    tilingData.mmTilingData.set_baseK(baseK);
    tilingData.mmTilingData.set_stepM(1);
    tilingData.mmTilingData.set_stepN(1);
    tilingData.mmTilingData.set_stepKa(STEPKA);
    tilingData.mmTilingData.set_stepKb(STEPKB);
    tilingData.mmTilingData.set_depthA1(DEPTHA1);
    tilingData.mmTilingData.set_depthB1(DEPTHB1);

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
    tilingData.set_groups(groups);
    tilingData.set_coreCount(aivNum);
    tilingData.set_singleLoopTask(singleLoopTask);
    tilingData.set_coreTaskCount(coreTaskCount);
    tilingData.set_bigCoreCount(bigCoreCount);
    tilingData.set_cubeTileTaskCount(cubeTileTaskCount);

    ADD_TILING_DATA(context, tilingData);

    size_t systemWorkspaceSize = ascendPlatformInfo.GetLibApiWorkSpaceSize();
    size_t usrWorkSpaceSize = 0;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    CHECK_NULLPTR(currentWorkspace);
    currentWorkspace[0] = systemWorkspaceSize + usrWorkSpaceSize;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForDeformableConv2dV2(gert::InferShapeContext* context)
{
    CHECK_NULLPTR(context);
    const gert::Shape* xShape = context->GetInputShape(INPUT_FEATURES_INDEX);
    const gert::Shape* offsetShape = context->GetInputShape(INPUT_OFFSET_INDEX);
    const gert::Shape* weightShape = context->GetInputShape(INPUT_WEIGHT_INDEX);
    if (xShape == nullptr || offsetShape == nullptr || weightShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    gert::Shape* xOffsetShape = context->GetOutputShape(OUTPUT_OFFSET_INDEX);
    if (xOffsetShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto attrsPtr = context->GetAttrs();
    CHECK_NULLPTR(attrsPtr);
    const auto* kernelSizePtr = attrsPtr->GetListInt(ATTR_KERNEL_DIM);
    CHECK_NULLPTR(kernelSizePtr);
    auto kernelSize = kernelSizePtr->GetData();

    int64_t B = xShape->GetDim(DIM_ZERO);
    int64_t Cin = xShape->GetDim(DIM_THREE);
    int64_t Hout = offsetShape->GetDim(DIM_ONE);
    int64_t Wout = offsetShape->GetDim(DIM_TWO);
    int64_t kh = kernelSize[0];
    int64_t kw = kernelSize[1];
    int64_t Cout = weightShape->GetDim(DIM_ZERO);

    *xOffsetShape = {B, Hout * Wout, kh * kw, Cin};
    *yShape = {B, Hout, Wout, Cout};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForDeformableConv2dV2(gert::InferDataTypeContext* context)
{
    CHECK_NULLPTR(context);
    const ge::DataType value_dtype = context->GetInputDataType(INPUT_FEATURES_INDEX);
    context->SetOutputDataType(OUTPUT_Y_INDEX, value_dtype);
    context->SetOutputDataType(OUTPUT_OFFSET_INDEX, value_dtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class DeformableConv2dV2 : public OpDef {
public:
    explicit DeformableConv2dV2(const char* name) : OpDef(name)
    {
        this->Input("inputFeatures")
            .ParamType(REQUIRED)
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

        this->Attr("kernel_size").ListInt();
        this->Attr("stride").ListInt();
        this->Attr("padding").ListInt();
        this->Attr("dilation").ListInt();
        this->Attr("groups").Int();
        this->Attr("deformable_groups").Int();
        this->Attr("modulated").Bool();
        this->Attr("with_bias").Bool();

        this->Output("outputFeatures")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        
        this->Output("offsetOutput")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForDeformableConv2dV2)
            .SetInferDataType(ge::InferDataTypeForDeformableConv2dV2);
        this->AICore().SetTiling(optiling::TilingForDeformableConv2dV2);
        this->AICore().AddConfig("ascend910_95");
    }
};

OP_ADD(DeformableConv2dV2);
} // namespace ops
