#include "subm_sparse_conv3d_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace ge;
using namespace std;
using namespace AscendC;
namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static int32_t GetCeilInt(int32_t value1, int32_t value2)
{
    if (value2 == 0) {
        return value1;
    }
    return static_cast<int32_t>((value1 + value2 - 1) / value2);
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SubmSparseConv3dTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    auto core_number = ascendplatformInfo.GetCoreNumAiv();
    uint32_t totalresult = context->GetInputTensor(0)->GetStorageShape().GetDim(0);
    auto feature_shape = context->GetInputTensor(0)->GetStorageShape();
    auto indices_shape = context->GetInputTensor(1)->GetStorageShape();
    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto kernel_size = attrsPtr->GetAttrPointer<gert::ContinuousVector>(0);
    auto kernel_size_data = reinterpret_cast<const int64_t*>(kernel_size->GetData());
    tiling.set_K0(kernel_size_data[0]);
    tiling.set_K1(kernel_size_data[1]);
    tiling.set_K2(kernel_size_data[2]);
    auto out_spatial_shape = attrsPtr->GetAttrPointer<gert::ContinuousVector>(2);
    auto out_spatial_shape_data = reinterpret_cast<const int64_t*>(out_spatial_shape->GetData());
    tiling.set_D(out_spatial_shape_data[0]);
    tiling.set_H(out_spatial_shape_data[1]);
    tiling.set_W(out_spatial_shape_data[2]);
    tiling.set_feature_map_size(out_spatial_shape_data[0] * out_spatial_shape_data[1]*out_spatial_shape_data[2]);
    auto out_channel = *(attrsPtr->GetAttrPointer<int32_t>(1));
    auto batch_size = *(attrsPtr->GetAttrPointer<int32_t>(3));
    int32_t core_data;
    int32_t core_used;
    int32_t core_last;
    core_data = GetCeilInt(totalresult, core_number);
    core_data = GetCeilInt(core_data, 64) * 64;
    core_used = GetCeilInt(totalresult, core_data);
    core_last = core_data;
    if (core_data == 0) {
        return ge::GRAPH_FAILED;
    }
    if (totalresult % core_data != 0) { core_last = totalresult % core_data;}
    uint64_t available_ub_size;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, available_ub_size);
    int32_t number = 20;
    int32_t total_kernel = kernel_size_data[0] * kernel_size_data[1] * kernel_size_data[2];
    available_ub_size = (available_ub_size - 20*1024 - total_kernel*6*4 - feature_shape.GetDim(1)*4) / number;
    available_ub_size = GetCeilInt(available_ub_size, 64) * 64;
    context->SetBlockDim(core_used);
    tiling.set_core_data(core_data);
    tiling.set_core_used(core_used);
    tiling.set_copy_loop(core_data / available_ub_size);
    tiling.set_copy_tail(core_data % available_ub_size);
    tiling.set_last_copy_loop(core_last / available_ub_size);
    tiling.set_last_copy_tail(core_last % available_ub_size);
    tiling.set_inchannel(feature_shape.GetDim(1));
    tiling.set_outchannel(out_channel);
    tiling.set_indices_number(indices_shape.GetDim(1));
    tiling.set_available_ub_size(available_ub_size);
    tiling.set_batch_size(batch_size);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 1;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto kernel_size = attrsPtr->GetAttrPointer<gert::ContinuousVector>(0);
    auto kernel_size_data = reinterpret_cast<const int64_t*>(kernel_size->GetData());
    const gert::Shape* indices_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (y_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* indices_out_shape = context->GetOutputShape(1);
    if (indices_out_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* indices_pair_shape = context->GetOutputShape(2);
    if (indices_pair_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto kernel_num = kernel_size_data[0] * kernel_size_data[1] * kernel_size_data[2];
    auto output_num = indices_shape->GetDim(0) * kernel_num;
    auto batch_size = *(attrsPtr->GetAttrPointer<int32_t>(3));
    auto out_channel = *(attrsPtr->GetAttrPointer<int32_t>(1));
    y_shape->SetDimNum(0);
    y_shape->AppendDim(output_num);
    y_shape->AppendDim(out_channel);
    indices_out_shape->SetDimNum(0);
    indices_out_shape->AppendDim(output_num);
    indices_pair_shape->SetDimNum(0);
    indices_pair_shape->AppendDim(output_num);
    indices_pair_shape->AppendDim(indices_shape->GetDim(1));
    return GRAPH_SUCCESS;
}
}


namespace ops {
class SubmSparseConv3d : public OpDef {
public:
    explicit SubmSparseConv3d(const char* name) : OpDef(name)
    {
        this->Input("feature")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("temp")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("feature_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("indices_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("indices_pair")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("kernel_size")
            .AttrType(REQUIRED)
            .ListInt();
        this->Attr("out_channel")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("outSpatialShape")
            .AttrType(REQUIRED)
            .ListInt();
        this->Attr("batch_size")
            .AttrType(REQUIRED)
            .Int();

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SubmSparseConv3d);
}