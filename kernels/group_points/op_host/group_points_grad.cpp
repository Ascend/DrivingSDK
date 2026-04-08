// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "group_points_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t SINGLE_INDICES = 1;
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t SIZE_OF_FP16 = 2;
constexpr uint32_t SIZE_OF_FP32 = 4;
constexpr uint32_t BLOCK_INT32 = 8;

constexpr size_t B_IDX = 0;
constexpr size_t C_IDX = 1;
constexpr size_t N_IDX = 2;
constexpr size_t NP_IDX = 3;
constexpr size_t NS_IDX = 4;

} // namespace
namespace optiling {

static ge::graphStatus TilingForGroupPointsGrad(gert::TilingContext* context)
{
    GroupPointsGradTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static uint32_t core_num = ascendcPlatform.GetCoreNumAiv();
    if (core_num == 0) {
        return ge::GRAPH_FAILED;
    }

    if (context->GetInputDesc(0) == nullptr || context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dtype = context->GetInputDesc(0)->GetDataType();
    if (ge::DT_FLOAT == dtype) {
        context->SetTilingKey(1);
    } else if (ge::DT_FLOAT16 == dtype) {
        context->SetTilingKey(2);
    } else {
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    auto getAttr = [attrs](size_t idx) -> int32_t {
        auto ptr = attrs->GetInt(idx);
        if (!ptr) {
            return -1;
        }
        return static_cast<int32_t>(*ptr);
    };
    auto b = getAttr(B_IDX);
    auto c = getAttr(C_IDX);
    auto n = getAttr(N_IDX);
    auto npoints = getAttr(NP_IDX);
    auto nsample = getAttr(NS_IDX);

    uint32_t alignNum = BYTE_BLOCK / SIZE_OF_FP32;
    if (dtype == ge::DT_FLOAT16) {
        alignNum = BYTE_BLOCK / SIZE_OF_FP16;
    }
    uint32_t cAligned = (c + alignNum - 1) / alignNum * alignNum;
    uint32_t indicesAligned = (SINGLE_INDICES + BLOCK_INT32 - 1) / BLOCK_INT32 * BLOCK_INT32;
    uint32_t average = b * npoints * nsample / core_num;
    uint32_t taskLast = b * npoints * nsample % core_num;
    uint32_t usedCoreNum = core_num;
    if (average == 0) {
        usedCoreNum = taskLast;
    }

    context->SetBlockDim(usedCoreNum);

    tiling.set_b(b);
    tiling.set_c(c);
    tiling.set_n(n);
    tiling.set_npoints(npoints);
    tiling.set_nsample(nsample);
    tiling.set_cAligned(cAligned);
    tiling.set_indicesAligned(indicesAligned);
    tiling.set_average(average);
    tiling.set_taskLast(taskLast);
    tiling.set_usedCoreNum(usedCoreNum);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling


namespace ge {
static ge::graphStatus InferShapeForGroupPointsGrad(gert::InferShapeContext* context)
{
    if (context->GetOutputShape(0) == nullptr|| context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    auto getAttr = [attrs](size_t idx) -> int32_t {
        auto ptr = attrs->GetInt(idx);
        if (!ptr) {
            return -1;
        }
        return static_cast<int32_t>(*ptr);
    };
    auto b = getAttr(B_IDX);
    auto c = getAttr(C_IDX);
    auto n = getAttr(N_IDX);

    gert::Shape* outShape = context->GetOutputShape(0);
    *outShape = {b * n, c};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForGroupPointsGrad(gert::InferDataTypeContext* context)
{
    const ge::DataType value_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, value_dtype);
    return GRAPH_SUCCESS;
}

} // namespace ge


namespace ops {
class GroupPointsGrad : public OpDef {
public:
    explicit GroupPointsGrad(const char* name) : OpDef(name)
    {
        this->Input("grad_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("grad_points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("b").AttrType(REQUIRED).Int();
        this->Attr("c").AttrType(REQUIRED).Int();
        this->Attr("n").AttrType(REQUIRED).Int();
        this->Attr("npoints").AttrType(REQUIRED).Int();
        this->Attr("nsample").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShapeForGroupPointsGrad).SetInferDataType(ge::InferDataTypeForGroupPointsGrad);

        this->AICore().SetTiling(optiling::TilingForGroupPointsGrad);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GroupPointsGrad);
} // namespace ops