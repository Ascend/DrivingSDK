/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "subm_sparse_conv3d_with_key_tiling.h"
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
    SubmSparseConv3dWithKeyTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    auto coreNumber = ascendplatformInfo.GetCoreNumAiv();
    uint32_t totalResult = context->GetInputTensor(1)->GetShapeSize();
    auto grad_shape = context->GetInputTensor(2)->GetStorageShape();
    auto valid_indices_shape = context->GetInputTensor(1)->GetStorageShape();
    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t inchannel = *(attrsPtr->GetAttrPointer<int32_t>(1));
    auto kernel_size = attrsPtr->GetAttrPointer<gert::ContinuousVector>(0);
    auto kernel_size_data = reinterpret_cast<const int64_t*>(kernel_size->GetData());
    int64_t kernel_num = kernel_size_data[0] * kernel_size_data[1] * kernel_size_data[2];
    tiling.set_K0(kernel_size_data[0]);
    tiling.set_K1(kernel_size_data[1]);
    tiling.set_K2(kernel_size_data[2]);
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
    int64_t outchannel = grad_shape.GetDim(1);
    int64_t number = outchannel + 2;
    int64_t valid_number = valid_indices_shape.GetDim(0);
    availableUbSize = (availableUbSize - 20*1024 - outchannel*70) / 20;
    availableUbSize = GetCeilInt(availableUbSize, 32) * 32;
    context->SetBlockDim(coreUsed);
    tiling.set_core_data(coreData);
    tiling.set_core_used(coreUsed);
    tiling.set_copy_loop(coreData / availableUbSize);
    tiling.set_copy_tail(coreData % availableUbSize);
    tiling.set_last_copy_loop(coreLast / availableUbSize);
    tiling.set_last_copy_tail(coreLast % availableUbSize);
    tiling.set_inchannel(outchannel);
    tiling.set_outchannel(outchannel);
    tiling.set_kernel_size(kernel_num);
    tiling.set_valid_number(valid_number);
    tiling.set_available_ub_size(availableUbSize);
    tiling.set_indices_number(grad_shape.GetDim(0));
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

static ge::graphStatus SubmSparseConv3dWithKeyInferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType x_dtype = context->GetInputDataType(2);
    context->SetOutputDataType(0, x_dtype);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class SubmSparseConv3dWithKey : public OpDef {
public:
    explicit SubmSparseConv3dWithKey(const char* name) : OpDef(name)
    {
        this->Input("outidx_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("valid_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("grad_out_features")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_out_features_iml2col")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("kernel_size")
            .AttrType(REQUIRED)
            .ListInt();
        this->Attr("in_channel")
            .AttrType(REQUIRED)
            .Int();

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::SubmSparseConv3dWithKeyInferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SubmSparseConv3dWithKey);
}
