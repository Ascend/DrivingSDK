#include "deformable_conv2d_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

using namespace matmul_tiling;

namespace optiling {
static ge::graphStatus TilingForDeformableConv2d(gert::TilingContext* context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto aicNum = ascendPlatformInfo.GetCoreNumAic();
    auto aivNum = ascendPlatformInfo.GetCoreNumAiv();
    if (aicNum == 0 || aivNum == 0) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(aicNum);

    auto xShape = context->GetInputShape(0)->GetStorageShape();      // n, cIn, hIn, wIn
    auto offsetShape = context->GetInputShape(3)->GetStorageShape(); // n, hOut, wOut, 2*kH*kW
    auto weightShape = context->GetInputShape(1)->GetStorageShape(); // kH, kW, cIn, cOut

    uint32_t n = xShape.GetDim(0);
    uint32_t cIn = xShape.GetDim(3);
    uint32_t hIn = xShape.GetDim(1);
    uint32_t wIn = xShape.GetDim(2);
    uint32_t cOut = weightShape.GetDim(0);
    uint32_t hOut = offsetShape.GetDim(1);
    uint32_t wOut = offsetShape.GetDim(2);

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto kernelSize = attrsPtr->GetListInt(0)->GetData();
    auto stride = attrsPtr->GetListInt(1)->GetData();
    auto padding = attrsPtr->GetListInt(2)->GetData();
    auto dilation = attrsPtr->GetListInt(3)->GetData();
    auto modulated = attrsPtr->GetBool(6);
    uint32_t kH = kernelSize[0];
    uint32_t kW = kernelSize[1];

    context->SetTilingKey(*modulated);

    DeformableConv2dTilingData tilingData;
    matmul_tiling::MatmulApiTiling mmTiling(ascendPlatformInfo);
    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mmTiling.SetBType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mmTiling.SetShape(wOut, cOut, kH * kW * cIn);
    mmTiling.SetOrgShape(wOut, cOut, kH * kW * cIn);
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
    tilingData.set_usedBlkNum(aivNum);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t systemWorkspaceSize = ascendPlatformInfo.GetLibApiWorkSpaceSize();
    size_t auxSize = 2 * kH * kW * wOut * sizeof(float);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = systemWorkspaceSize + auxSize;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
namespace ge {
static ge::graphStatus InferShapeForDeformableConv2d(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* offsetShape = context->GetInputShape(1);
    const gert::Shape* weightShape = context->GetInputShape(2);
    if (xShape == nullptr || offsetShape == nullptr || weightShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* xOffsetShape = context->GetOutputShape(0);
    gert::Shape* yShape = context->GetOutputShape(1);
    if (xOffsetShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t B = xShape->GetDim(0);
    int64_t Hin = xShape->GetDim(1);
    int64_t Win = xShape->GetDim(2);
    int64_t Cin = xShape->GetDim(3);
    int64_t Hout = offsetShape->GetDim(1);
    int64_t Wout = offsetShape->GetDim(2);
    int64_t kh = weightShape->GetDim(0);
    int64_t kw = weightShape->GetDim(1);
    int64_t Cout = weightShape->GetDim(3);

    *xOffsetShape = {B, Hin * Win, kh * kw, Cin};
    *yShape = {B, Hout, Wout, Cout};
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeForDeformableConv2d(gert::InferDataTypeContext* context)
{
    const ge::DataType value_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, value_dtype);
    context->SetOutputDataType(1, value_dtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class DeformableConv2d : public OpDef {
public:
    explicit DeformableConv2d(const char* name) : OpDef(name)
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

        this->Attr("kernel_size").ListInt();
        this->Attr("stride").ListInt();
        this->Attr("padding").ListInt();
        this->Attr("dilation").ListInt();
        this->Attr("groups").Int();
        this->Attr("deformable_groups").Int();
        this->Attr("modulated").Bool();
        this->Attr("with_bias").Bool(); // false

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

        this->SetInferShape(ge::InferShapeForDeformableConv2d).SetInferDataType(ge::InferDataTypeForDeformableConv2d);
        this->AICore().SetTiling(optiling::TilingForDeformableConv2d);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(DeformableConv2d);
} // namespace ops
