#include "ge/utils.h"
#include "batch_matmul_vector_tiling.h"
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
    BatchMatmulVectorTilingData tiling;
    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    auto core_number = ascendplatformInfo.GetCoreNumAiv();
    CHECK_NULLPTR(context->GetInputTensor(0));
    uint32_t totalresult = context->GetInputTensor(0)->GetShapeSize();
    auto projection_mat_shape = context->GetInputTensor(0)->GetStorageShape();
    auto dimnum = projection_mat_shape.GetDimNum();
    if (dimnum < 3) {
        return ge::GRAPH_FAILED;
    }
    auto projection_matrix_dim4 = projection_mat_shape.GetDim(dimnum - 2);
    auto projection_matrix_dim5 = projection_mat_shape.GetDim(dimnum - 1);
    uint32_t ptstotal = context->GetInputTensor(1)->GetShapeSize();
    if (projection_matrix_dim5 == 0) {
        return ge::GRAPH_FAILED;
    }
    auto batch_size = totalresult / projection_matrix_dim5;
    int32_t core_data;
    int32_t core_used;
    int32_t core_last;
    core_data = GetCeilInt(batch_size, core_number);
    core_data = GetCeilInt(core_data, 64) * 64;
    core_used = GetCeilInt(batch_size, core_data);
    core_last = core_data;
    if (core_data == 0) {
        return ge::GRAPH_FAILED;
    }
    if (batch_size % core_data != 0) { core_last = batch_size % core_data;}
    uint64_t available_ub_size;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, available_ub_size);
    int32_t number = 24*4;
    available_ub_size = (available_ub_size) / number;
    available_ub_size = GetCeilInt(available_ub_size, 64) * 64;
    context->SetBlockDim(core_used);
    tiling.set_core_data(core_data);
    tiling.set_core_used(core_used);
    tiling.set_copy_loop(core_data / available_ub_size);
    tiling.set_copy_tail(core_data % available_ub_size);
    tiling.set_last_copy_loop(core_last / available_ub_size);
    tiling.set_last_copy_tail(core_last % available_ub_size);
    tiling.set_available_ub_size(available_ub_size);
    tiling.set_totalresult(totalresult);
    tiling.set_ptstotal(ptstotal);
    tiling.set_dim4(projection_matrix_dim4);
    tiling.set_dim5(projection_matrix_dim5);
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
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (y_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* indices_out_shape = context->GetOutputShape(1);
    if (indices_out_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
   
    return GRAPH_SUCCESS;
}
}


namespace ops {
class BatchMatmulVector : public OpDef {
public:
    explicit BatchMatmulVector(const char* name) : OpDef(name)
    {
        this->Input("projection_mat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("pts_extend")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("point_2d")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(BatchMatmulVector);
}