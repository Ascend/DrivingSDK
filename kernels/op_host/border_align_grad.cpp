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
#include "border_align_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
using namespace std;

namespace {
constexpr uint32_t SINGLE_INDICES = 1;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t SIZE_OF_FP16 = 2;
constexpr uint32_t SIZE_OF_FP32 = 4;
constexpr uint32_t BLOCK_INT32 = BLOCK_SIZE / SIZE_OF_FP32;


constexpr size_t C_IDX = 0;
constexpr size_t BS_IDX = 1;
constexpr size_t H_IDX = 2;
constexpr size_t W_IDX = 3;
constexpr size_t PS_IDX = 4;
constexpr size_t BAS_IDX = 5;
} // namespace

namespace optiling {
static ge::graphStatus TilingForBorderAlignGrad(gert::TilingContext* context)
{
    BorderAlignGradTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    if (context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr || context->GetInputShape(2) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    if (context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto getAttr = [attrs](size_t idx) -> int32_t {
        auto ptr = attrs->GetInt(idx);
        if (!ptr) {
            return ge::GRAPH_FAILED;
        }
        return static_cast<int32_t>(*ptr);
    };
    auto channels = getAttr(C_IDX);
    auto boxSize = getAttr(BS_IDX);
    auto height = getAttr(H_IDX);
    auto width = getAttr(W_IDX);
    auto poolSize = getAttr(PS_IDX);
    auto batchSize = getAttr(BAS_IDX);

    auto inputInf = context->GetInputDesc(0);
    if (inputInf == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dtype = inputInf->GetDataType();

    int64_t coreCompNum = batchSize * channels * boxSize / coreNum;
    int64_t taskLast = batchSize * channels * boxSize % coreNum;

    context->SetBlockDim(coreNum);
    tiling.set_channels(channels);
    tiling.set_boxSize(boxSize);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_poolSize(poolSize);
    tiling.set_batchSize(batchSize);
    tiling.set_coreCompNum(coreCompNum);
    tiling.set_taskLast(taskLast);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling


namespace ge {
static ge::graphStatus InferShapeForBorderAlignGrad(gert::InferShapeContext* context)
{
    if (context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr || context->GetInputShape(2) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    if (context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    auto getAttr = [attrs](size_t idx) -> int32_t {
        auto ptr = attrs->GetInt(idx);
        if (!ptr) {
            return ge::GRAPH_FAILED;
        }
        return static_cast<int32_t>(*ptr);
    };
    auto channels = getAttr(C_IDX);
    auto height = getAttr(H_IDX);
    auto width = getAttr(W_IDX);
    auto batchSize = getAttr(BAS_IDX);
    
    gert::Shape* outShape = context->GetOutputShape(0);
    *outShape = {batchSize, channels * 4, height, width};
    return GRAPH_SUCCESS;
}


static ge::graphStatus InferDataTypeForBorderAlignGrad(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
} // namespace ge


namespace ops {
class BorderAlignGrad : public OpDef {
public:
    explicit BorderAlignGrad(const char* name) : OpDef(name)
    {
        this->Input("gradOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("boxes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("argmaxIdx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("channels").AttrType(REQUIRED).Int();
        this->Attr("boxSize").AttrType(REQUIRED).Int();
        this->Attr("height").AttrType(REQUIRED).Int();
        this->Attr("width").AttrType(REQUIRED).Int();
        this->Attr("poolSize").AttrType(REQUIRED).Int();
        this->Attr("batchSize").AttrType(REQUIRED).Int();

        this->Output("gradInput")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForBorderAlignGrad).SetInferDataType(ge::InferDataTypeForBorderAlignGrad);

        this->AICore().SetTiling(optiling::TilingForBorderAlignGrad);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(BorderAlignGrad);
} // namespace ops