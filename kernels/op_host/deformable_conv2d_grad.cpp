#include "deformable_conv2d_grad_tiling.h"
#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
ge::graphStatus TilingForDeformableConv2dGrad(gert::TilingContext* context)
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
    auto xShape = xShapePtr->GetStorageShape();           // n, cIn, hIn, wIn
    auto offsetShape = offsetShapePtr->GetStorageShape(); // n, hOut, wOut, 2*kH*kW
    auto weightShape = weightShapePtr->GetStorageShape(); // kH, kW, cIn, cOut

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
    uint64_t kH = kernelSize[0]; // NOTE: kernelSize[0]
    uint64_t kW = kernelSize[1];

    context->SetTilingKey(*modulatedPtr);

    DeformableConv2dGradTilingData tilingData;
    matmul_tiling::MatmulApiTiling mm0Tiling(ascendPlatformInfo), mm1Tiling(ascendPlatformInfo);
    mm0Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
    mm0Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm0Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm0Tiling.SetShape(wOut, kH * kW * cIn / groups, cOut / groups);
    mm0Tiling.SetOrgShape(wOut, kH * kW * cIn / groups, cOut / groups);
    mm0Tiling.SetBias(false);
    mm0Tiling.SetBufferSpace(-1, -1, -1);
    if (mm0Tiling.GetTiling(tilingData.mm0TilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    mm1Tiling.SetAType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm1Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm1Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm1Tiling.SetShape(cOut / groups, kH * kW * cIn / groups, wOut);
    mm1Tiling.SetOrgShape(cOut / groups, kH * kW * cIn / groups, wOut);
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
    tilingData.set_usedBlkNum(aivNum);

    ADD_TILING_DATA(context, tilingData);

    size_t systemWorkspaceSize = ascendPlatformInfo.GetLibApiWorkSpaceSize();
    size_t auxSize = 2 * kH * kW * wOut * sizeof(float);
    size_t gradOffsetOutputSize = n * hOut * wOut * kH * kW * cIn * sizeof(float);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    CHECK_NULLPTR(currentWorkspace);
    currentWorkspace[0] = systemWorkspaceSize + auxSize + gradOffsetOutputSize;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class DeformableConv2dGrad : public OpDef {
public:
    explicit DeformableConv2dGrad(const char* name) : OpDef(name)
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
        this->Input("offset_output")
            .ParamType(REQUIRED)
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
        this->Attr("groups").Int();            // 1
        this->Attr("deformable_groups").Int(); // 1
        this->Attr("modulated").Bool();        // true
        this->Attr("with_bias").Bool();        // false

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

        this->AICore().SetTiling(optiling::TilingForDeformableConv2dGrad);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(DeformableConv2dGrad);
} // namespace ops
