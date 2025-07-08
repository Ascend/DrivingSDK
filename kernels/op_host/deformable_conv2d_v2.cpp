#include "deformable_conv2d_v2_tiling.h"
#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr uint8_t INPUT_X_INDEX = 0;
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
} // namespace

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

    const auto xShapePtr = context->GetInputShape(INPUT_X_INDEX);
    const auto offsetShapePtr = context->GetInputShape(INPUT_OFFSET_INDEX);
    const auto weightShapePtr = context->GetInputShape(INPUT_WEIGHT_INDEX);
    CHECK_NULLPTR(xShapePtr);
    CHECK_NULLPTR(offsetShapePtr);
    CHECK_NULLPTR(weightShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    auto offsetShape = offsetShapePtr->GetStorageShape();
    auto weightShape = weightShapePtr->GetStorageShape();

    uint64_t n = xShape.GetDim(DIM_ZERO);
    uint64_t hIn = xShape.GetDim(DIM_ONE);
    uint64_t wIn = xShape.GetDim(DIM_TWO);
    uint64_t cIn = xShape.GetDim(DIM_THREE);
    uint64_t hOut = offsetShape.GetDim(DIM_ONE);
    uint64_t wOut = offsetShape.GetDim(DIM_TWO);
    uint64_t cOut = weightShape.GetDim(DIM_ZERO);

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
    auto groups = *groupsPtr;
    uint64_t kH = kernelSize[0];
    uint64_t kW = kernelSize[1];

    // kernel tiling
    uint32_t cubeTileTaskCount = 128;
    uint32_t totalTasks = n * hOut * wOut;
    uint32_t avgTasks = totalTasks / aivNum;
    uint32_t remainTasks = totalTasks % aivNum;

    context->SetTilingKey(*modulatedPtr);

    DeformableConv2dV2TilingData tilingData;
    matmul_tiling::MatmulApiTiling mmTiling(ascendPlatformInfo);
    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mmTiling.SetBType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mmTiling.SetShape(cubeTileTaskCount, cOut, kH * kW * cIn);
    mmTiling.SetOrgShape(cubeTileTaskCount, cOut, kH * kW * cIn);
    mmTiling.SetBias(false);
    mmTiling.SetBufferSpace(-1, -1, -1);
    if (mmTiling.GetTiling(tilingData.mmTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

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
    tilingData.set_coreCount(aivNum);
    tilingData.set_avgTasks(avgTasks);
    tilingData.set_bigCoreCount(remainTasks);
    tilingData.set_cubeTileTaskCount(cubeTileTaskCount);

    ADD_TILING_DATA(context, tilingData);

    size_t systemWorkspaceSize = ascendPlatformInfo.GetLibApiWorkSpaceSize();
    size_t usrWorkSpaceSize = n * hOut * wOut * kH * kW * cIn * sizeof(float);
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
    const gert::Shape* xShape = context->GetInputShape(INPUT_X_INDEX);
    const gert::Shape* offsetShape = context->GetInputShape(INPUT_OFFSET_INDEX);
    const gert::Shape* weightShape = context->GetInputShape(INPUT_WEIGHT_INDEX);
    if (xShape == nullptr || offsetShape == nullptr || weightShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* xOffsetShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_OFFSET_INDEX);
    if (xOffsetShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto attrsPtr = context->GetAttrs();
    CHECK_NULLPTR(attrsPtr);
    const auto* kernelSizePtr = attrsPtr->GetListInt(ATTR_KERNEL_DIM);
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
    const ge::DataType value_dtype = context->GetInputDataType(INPUT_X_INDEX);
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
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("kernel_size").ListInt();
        this->Attr("stride").ListInt();
        this->Attr("padding").ListInt();
        this->Attr("dilation").ListInt();
        this->Attr("groups").Int();
        this->Attr("deformable_groups").Int();
        this->Attr("modulated").Bool();
        this->Attr("with_bias").Bool();

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("offset_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForDeformableConv2dV2)
            .SetInferDataType(ge::InferDataTypeForDeformableConv2dV2);
        this->AICore().SetTiling(optiling::TilingForDeformableConv2dV2);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(DeformableConv2dV2);
} // namespace ops
