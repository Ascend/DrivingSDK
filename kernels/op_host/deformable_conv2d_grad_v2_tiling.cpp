#include "deformable_conv2d_grad_v2_tiling.h"
#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint8_t INPUT_X_IDX = 0;
constexpr uint8_t INPUT_WEIGHT_IDX = 1;
constexpr uint8_t INPUT_BIAS_IDX = 2;
constexpr uint8_t INPUT_OFFSET_IDX = 3;
constexpr uint8_t INPUT_MASK_IDX = 4;

constexpr uint8_t OUTPUT_X_GRAD_IDX = 0;
constexpr uint8_t OUTPUT_WEIGHT_GRAD_IDX = 1;
constexpr uint8_t OUTPUT_BIAS_GRAD_IDX = 2;
constexpr uint8_t OUTPUT_OFFSET_GRAD_IDX = 3;
constexpr uint8_t OUTPUT_MASK_GRAD_IDX = 4;

constexpr uint8_t DIM_IDX_ZERO = 0;
constexpr uint8_t DIM_IDX_ONE = 1;
constexpr uint8_t DIM_IDX_TWO = 2;
constexpr uint8_t DIM_IDX_THREE = 3;
} // namespace


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
    CHECK_NULLPTR(xShapePtr);
    CHECK_NULLPTR(offsetShapePtr);
    CHECK_NULLPTR(weightShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    auto offsetShape = offsetShapePtr->GetStorageShape();
    auto weightShape = weightShapePtr->GetStorageShape();

    uint64_t n = xShape.GetDim(0);
    uint64_t cIn = xShape.GetDim(3);
    uint64_t hIn = xShape.GetDim(1);
    uint64_t wIn = xShape.GetDim(2);
    uint64_t cOut = weightShape.GetDim(0);
    uint64_t hOut = offsetShape.GetDim(1);
    uint64_t wOut = offsetShape.GetDim(2);

    auto attrsPtr = context->GetAttrs();
    CHECK_NULLPTR(attrsPtr);
    const auto* kernelSizePtr = attrsPtr->GetListInt(0);
    const auto* stridePtr = attrsPtr->GetListInt(1);
    const auto* paddingPtr = attrsPtr->GetListInt(2);
    const auto* dilationPtr = attrsPtr->GetListInt(3);
    const auto* groupsPtr = attrsPtr->GetInt(4);
    const auto* modulatedPtr = attrsPtr->GetBool(6);
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
    uint32_t cubeTileTaskCount = 64;

    context->SetTilingKey(*modulatedPtr);

    DeformableConv2dGradV2TilingData tilingData;

    matmul_tiling::MatmulApiTiling mm0Tiling(ascendPlatformInfo);
    matmul_tiling::MatmulApiTiling mm1Tiling(ascendPlatformInfo);
    mm0Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm0Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm0Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm0Tiling.SetOrgShape(cubeTileTaskCount, kH * kW * cIn, cOut);
    mm0Tiling.SetShape(-1, -1, -1);
    mm0Tiling.SetBias(false);
    mm0Tiling.SetBufferSpace(-1, -1, -1);
    if (mm0Tiling.GetTiling(tilingData.mm0TilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    mm1Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
    mm1Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm1Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm1Tiling.SetOrgShape(cOut, kH * kW * cIn, cubeTileTaskCount);
    mm1Tiling.SetShape(-1, -1, -1);
    mm1Tiling.SetBias(false);
    mm1Tiling.SetBufferSpace(-1, -1, -1);
    if (mm1Tiling.GetTiling(tilingData.mm1TilingData) == -1) {
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
    tilingData.set_groups(*groupsPtr);
    tilingData.set_coreCount(aivNum);
    tilingData.set_cubeTileTaskCount(cubeTileTaskCount);
    
    ADD_TILING_DATA(context, tilingData);

    size_t systemWorkspaceSize = ascendPlatformInfo.GetLibApiWorkSpaceSize();
    size_t usrWorkSpaceSize = n * hOut * wOut * kH * kW * cIn * 2 * sizeof(float);
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
    const gert::Shape* xShape = context->GetInputShape(INPUT_X_IDX);
    const gert::Shape* weightShape = context->GetInputShape(INPUT_WEIGHT_IDX);
    const gert::Shape* offsetShape = context->GetInputShape(INPUT_OFFSET_IDX);
    const gert::Shape* maskShape = context->GetInputShape(INPUT_MASK_IDX);
    CHECK_NULLPTR(xShape);
    CHECK_NULLPTR(weightShape);
    CHECK_NULLPTR(offsetShape);
    CHECK_NULLPTR(maskShape);

    int64_t batchSize = xShape->GetDim(DIM_IDX_ZERO);
    int64_t hIn = xShape->GetDim(DIM_IDX_ONE);
    int64_t wIn = xShape->GetDim(DIM_IDX_TWO);
    int64_t inChannels = xShape->GetDim(DIM_IDX_THREE);

    gert::Shape* xGradShape = context->GetOutputShape(OUTPUT_X_GRAD_IDX);
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
}

static ge::graphStatus InferDataType4DeformableConv2dGradV2(gert::InferDataTypeContext *context)
{
    const ge::DataType featureDtype = context->GetInputDataType(INPUT_X_IDX);
    context->SetOutputDataType(OUTPUT_X_GRAD_IDX, featureDtype);
    context->SetOutputDataType(OUTPUT_WEIGHT_GRAD_IDX, featureDtype);
    context->SetOutputDataType(OUTPUT_OFFSET_GRAD_IDX, featureDtype);
    context->SetOutputDataType(OUTPUT_MASK_GRAD_IDX, featureDtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class DeformableConv2dGradV2 : public OpDef {
public:
    explicit DeformableConv2dGradV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
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
        this->Input("grad_y")
            .ParamType(REQUIRED)
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

        this->Output("grad_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        
        this->SetInferShape(ge::InferShape4DeformableConv2dGradV2)
            .SetInferDataType(ge::InferDataType4DeformableConv2dGradV2);
        this->AICore().SetTiling(optiling::TilingForDeformableConv2dGradV2);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(DeformableConv2dGradV2);
} // namespace ops
