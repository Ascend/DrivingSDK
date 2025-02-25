#include "deformable_conv2d_tiling.h"
#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
using namespace matmul_tiling;

namespace optiling {
static ge::graphStatus TilingForDeformableConv2d(gert::TilingContext* context)
{
    CHECK_NULLPTR(context);
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

    const gert::StorageShape* xShapePtr = context->GetInputShape(0);
    const gert::StorageShape* offsetShapePtr = context->GetInputShape(3);
    const gert::StorageShape* weightShapePtr = context->GetInputShape(1);
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
    CHECK_NULLPTR(attrsPtr)

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

    context->SetTilingKey(*modulatedPtr);

    DeformableConv2dTilingData tilingData;
    matmul_tiling::MatmulApiTiling mmTiling(ascendPlatformInfo);
    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mmTiling.SetBType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mmTiling.SetShape(cOut / groups, wOut, kH * kW * cIn / groups);
    mmTiling.SetOrgShape(cOut / groups, wOut, kH * kW * cIn / groups);
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
    tilingData.set_usedBlkNum(aivNum);

    ADD_TILING_DATA(context, tilingData);

    size_t systemWorkspaceSize = ascendPlatformInfo.GetLibApiWorkSpaceSize();
    size_t auxSize = 2 * kH * kW * wOut * sizeof(float);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    CHECK_NULLPTR(currentWorkspace);
    currentWorkspace[0] = systemWorkspaceSize + auxSize;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
namespace ge {
static ge::graphStatus InferShapeForDeformableConv2d(gert::InferShapeContext* context)
{
    CHECK_NULLPTR(context);
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
    CHECK_NULLPTR(context)
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
