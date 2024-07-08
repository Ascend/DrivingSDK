/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "add_relu_tiling.h"
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
    AddReluTilingData tiling;
    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    auto CoreNumber = ascendplatformInfo.GetCoreNumAiv();
    uint32_t TotalResult = context->GetInputTensor(0)->GetShapeSize();
    int32_t CoreData;
    int32_t CoreUsed;
    int32_t CoreLast;
    CoreData = GetCeilInt(TotalResult, CoreNumber);
    CoreData = GetCeilInt(CoreData, 64) * 64;
    CoreUsed = GetCeilInt(totalresult, CoreData);
    CoreLast = CoreData;
    if (CoreData == 0) {
        return ge::GRAPH_FAILED;
    }
    if (TotalResult % CoreData != 0) { CoreLast = TotalResult % CoreData;}
    uint64_t AvailableUbSize;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, AvailableUbSize);
    AvailableUbSize = (AvailableUbSize - 20*1024) / 12;
    AvailableUbSize = GetCeilInt(AvailableUbSize, 32) * 32;
    context->SetBlockDim(coreUsed);
    tiling.set_core_data(CoreData);
    tiling.set_core_used(coreUsed);
    tiling.set_copy_loop(CoreData / AvailableUbSize);
    tiling.set_copy_tail(CoreData % AvailableUbSize);
    tiling.set_last_copy_loop(CoreLast / AvailableUbSize);
    tiling.set_last_copy_tail(CoreLast % AvailableUbSize);
    tiling.set_box_number(totalresult);
    tiling.set_available_ub_size(AvailableUbSize);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    return GRAPH_SUCCESS;
}
}


namespace ops {
class AddRelu : public OpDef {
public:
    explicit AddRelu(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910c");
    }
};

OP_ADD(AddRelu);
}
