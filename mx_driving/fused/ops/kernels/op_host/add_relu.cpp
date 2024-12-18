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
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    auto coreNumber = ascendplatformInfo.GetCoreNumAiv();
    uint32_t totalResult = context->GetInputTensor(0)->GetShapeSize();
    int32_t coreData;
    int32_t coreUsed;
    int32_t coreLast;
    coreData = GetCeilInt(totalResult, coreNumber);
    coreData = GetCeilInt(coreData, 64) * 64;
    coreUsed = GetCeilInt(totalResult, coreData);
    coreLast = coreData;
    if (coreData == 0) {
        return ge::GRAPH_FAILED;
    }
    if (totalResult % coreData != 0) { coreLast = totalResult % coreData;}
    uint64_t availableUbSize;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, availableUbSize);
    availableUbSize = (availableUbSize - 20*1024) / 12;
    availableUbSize = GetCeilInt(availableUbSize, 32) * 32;
    context->SetBlockDim(coreUsed);
    tiling.set_core_data(coreData);
    tiling.set_core_used(coreUsed);
    tiling.set_copy_loop(coreData / availableUbSize);
    tiling.set_copy_tail(coreData % availableUbSize);
    tiling.set_last_copy_loop(coreLast / availableUbSize);
    tiling.set_last_copy_tail(coreLast % availableUbSize);
    tiling.set_box_number(totalResult);
    tiling.set_available_ub_size(availableUbSize);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
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

static ge::graphStatus AddReluInferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType x_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, x_dtype);
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
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::AddReluInferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(AddRelu);
}
