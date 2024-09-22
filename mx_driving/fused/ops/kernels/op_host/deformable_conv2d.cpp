#include "deformable_conv2d_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

using namespace matmul_tiling;

namespace {
    constexpr uint32_t BYTE_BLOCK = 32;
    constexpr uint32_t SIZE_OF_FP16 = 2;
    constexpr uint32_t SIZE_OF_FP32 = 4;
}
namespace optiling {
static int32_t GetCeilInt(int32_t value1, int32_t value2)
{
    if (value2 == 0) {
    return value1;
    }
    return static_cast<int32_t>((value1 + value2 - 1) / value2);
}

static ge::graphStatus TilingForDeformableConv2d(gert::TilingContext* context)
{
    DeformableConv2dTilingData tiling;
    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    auto aicNum = ascendplatformInfo.GetCoreNumAic();
    auto aivNum = ascendplatformInfo.GetCoreNumAiv();
    if (aicNum == 0 || aivNum == 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t availableUbSize = ubSize / 12;
    auto x_shape = context->GetInputShape(0)->GetStorageShape(); // x in shape of (B, Hin, Win, Cin)
    auto offset_shape = context->GetInputShape(1)->GetStorageShape(); // offset in shape of (B, Hout, Wout, 2*kh*kw)
    auto weight_shape = context->GetInputShape(2)->GetStorageShape(); // weight in shape of (kh, kw, Cin, Cout)
    auto x_offset_shape = context->GetOutputShape(0)->GetStorageShape(); // xoffset in shape of (B, Hout*Wout, kh*kw, Cin)
    uint32_t xSize = x_shape.GetShapeSize();
    uint32_t weightSize = weight_shape.GetShapeSize();
    uint32_t Hin = x_shape.GetDim(1);
    uint32_t Win = x_shape.GetDim(2);
    uint32_t B = offset_shape.GetDim(0);
    uint32_t Hout = offset_shape.GetDim(1);
    uint32_t Wout = offset_shape.GetDim(2);
    uint32_t kh = weight_shape.GetDim(0);
    uint32_t kw = weight_shape.GetDim(1);
    uint32_t Cin = weight_shape.GetDim(2);
    uint32_t Cout = weight_shape.GetDim(3);
    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto kernelPtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(0);
    auto stridePtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(1);
    auto paddingPtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(2);
    uint32_t totalTask = B * Hout * Wout;
    uint32_t coreAvgTask = totalTask / aivNum;
    uint32_t mainCoreNum = totalTask % aivNum;
    uint32_t usedCoreNum = aivNum;
    if (coreAvgTask == 0) {
        usedCoreNum = mainCoreNum;
    }
    auto tensorDesc = context->GetInputDesc(0);
    if (tensorDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t alignNum = BYTE_BLOCK / SIZE_OF_FP32;
    uint32_t availableUbElem = availableUbSize / SIZE_OF_FP32;
    uint32_t offsetUnit = 2 * kh * kw;
    uint32_t xOffsetUnit = kh * kw * Cin;
    uint32_t offsetAligned = GetCeilInt(offsetUnit, alignNum) * alignNum;
    uint32_t xOffsetAligned = GetCeilInt(xOffsetUnit, alignNum) * alignNum;
    uint32_t cInAligned = GetCeilInt(Cin, alignNum) * alignNum;
    uint32_t taskSingleLoop = ((availableUbElem / offsetAligned) / 128) * 128;
    context->SetBlockDim(aicNum);
    tiling.set_c_in(Cin);
    tiling.set_h_in(Hin);
    tiling.set_w_in(Win);
    tiling.set_c_out(Cout);
    tiling.set_h_out(Hout);
    tiling.set_w_out(Wout);
    tiling.set_x_size(xSize);
    tiling.set_weight_size(weightSize);
    tiling.set_x_offset_unit(xOffsetUnit);
    tiling.set_c_in_aligned(cInAligned);
    tiling.set_use_core_num(usedCoreNum);
    tiling.set_core_avg_task(coreAvgTask);
    tiling.set_main_core_num(mainCoreNum);
    tiling.set_task_single_loop(taskSingleLoop);
    int32_t M = aivNum * taskSingleLoop;
    int32_t N = Cout;
    int32_t K = kh * kw * Cin;
    MultiCoreMatmulTiling cubeTiling(ascendplatformInfo);
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetDim(aivNum);
    cubeTiling.SetOrgShape(M, N, K); // 原始完整的形状M、N、Ka、Kb
    cubeTiling.SetShape(M, N, K); // 单次计算的形状singleM、singleN、singleK
    cubeTiling.SetBufferSpace(-1, -1, -1);
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = ascendplatformInfo.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
}
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

    *xOffsetShape = {B, Hin*Win, kh*kw, Cin};
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
}

namespace ops {
class DeformableConv2d : public OpDef {
public:
    explicit DeformableConv2d(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("x_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("kernel_size").ListInt();
        this->Attr("stride").ListInt();
        this->Attr("padding").ListInt();
        this->Attr("dilation").ListInt();
        this->Attr("groups").Int();
        this->Attr("deformable_groups").Int();
        this->Attr("modulated").Bool();
        this->SetInferShape(ge::InferShapeForDeformableConv2d).SetInferDataType(ge::InferDataTypeForDeformableConv2d);
        this->AICore().SetTiling(optiling::TilingForDeformableConv2d);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(DeformableConv2d);
}