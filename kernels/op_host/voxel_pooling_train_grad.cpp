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
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "voxel_pooling_train_grad_tiling.h"

namespace {
constexpr uint32_t INDICES = 3;
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t SIZE_OF_FP32 = 4;
constexpr uint32_t BLOCK_INT32 = 8;

constexpr size_t B_IDX = 0;
constexpr size_t N_IDX = 1;
constexpr size_t C_IDX = 2;
constexpr size_t H_IDX = 3;
constexpr size_t W_IDX = 4;

} // namespace
namespace optiling {

static ge::graphStatus TilingForVoxelPoolingTrainGrad(gert::TilingContext* context)
{
    VoxelPoolingTrainGradTilingData tiling;
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

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto getAttr = [attrs](size_t idx) -> int32_t {
        auto ptr = attrs->GetInt(idx);
        if (!ptr) {
            return -1;
        }
        return static_cast<int32_t>(*ptr);
    };

    auto batch_size = getAttr(B_IDX);
    auto num_points = getAttr(N_IDX);
    auto num_channels = getAttr(C_IDX);
    auto h = getAttr(H_IDX);
    auto w = getAttr(W_IDX);

    uint32_t alignNum = BYTE_BLOCK / SIZE_OF_FP32; // 8

    uint32_t numChannelsAligned = (num_channels + alignNum - 1) / alignNum * alignNum;
    uint32_t indicesAligned = (INDICES + BLOCK_INT32 - 1) / BLOCK_INT32 * BLOCK_INT32;
    uint32_t average = batch_size * num_points / core_num;
    uint32_t taskLast = batch_size * num_points % core_num;
    uint32_t usedCoreNum = core_num;
    if (average == 0) {
        usedCoreNum = taskLast;
    }

    context->SetBlockDim(usedCoreNum);

    tiling.set_batchSize(batch_size);
    tiling.set_numPoints(num_points);
    tiling.set_numChannels(num_channels);
    tiling.set_h(h);
    tiling.set_w(w);
    tiling.set_numChannelsAligned(numChannelsAligned);
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
static ge::graphStatus InferShapeForVoxelPoolingTrainGrad(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto getAttr = [attrs](size_t idx) -> int32_t {
        auto ptr = attrs->GetInt(idx);
        if (!ptr) {
            return -1;
        }
        return static_cast<int32_t>(*ptr);
    };
    auto batchSize = getAttr(B_IDX);
    auto numPoints = getAttr(N_IDX);
    auto numChannels = getAttr(C_IDX);

    if (context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* grad_features = context->GetOutputShape(0);
    grad_features->SetDimNum(0);
    grad_features->AppendDim(batchSize);
    grad_features->AppendDim(numPoints);
    grad_features->AppendDim(numChannels);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForVoxelPoolingTrainGrad(gert::InferDataTypeContext* context)
{
    const ge::DataType value_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, value_dtype);
    return GRAPH_SUCCESS;
}

} // namespace ge


namespace ops {
class VoxelPoolingTrainGrad : public OpDef {
public:
    explicit VoxelPoolingTrainGrad(const char* name) : OpDef(name)
    {
        this->Input("grad_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("pos_memo")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_features")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("batch_size").Int();
        this->Attr("num_points").Int();
        this->Attr("num_channels").Int();
        this->Attr("h").Int();
        this->Attr("w").Int();

        this->SetInferShape(ge::InferShapeForVoxelPoolingTrainGrad)
            .SetInferDataType(ge::InferDataTypeForVoxelPoolingTrainGrad);

        this->AICore().SetTiling(optiling::TilingForVoxelPoolingTrainGrad);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(VoxelPoolingTrainGrad);
} // namespace ops