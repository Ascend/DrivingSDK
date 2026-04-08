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
#include "common.h"
#include "group_points_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace {
constexpr uint32_t SINGLE_INDICES = 1;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t SIZE_OF_FP16 = 2;
constexpr uint32_t SIZE_OF_FP32 = 4;
constexpr uint32_t SIZE_OF_INT32 = 4;
constexpr uint32_t BLOCK_INT32 = BLOCK_SIZE / SIZE_OF_FP32;
constexpr uint32_t MIN_CORE_TASK = 64;
constexpr uint32_t UB_TASK_BLOCK = BLOCK_SIZE / SIZE_OF_INT32;
constexpr uint64_t RPC_WORKSIZE = 20 * 1024;
constexpr uint64_t MAX_COPY_BLOCK_COUNT = 4095;

constexpr size_t B_IDX = 0;
constexpr size_t C_IDX = 1;
constexpr size_t N_IDX = 2;
constexpr size_t NP_IDX = 3;
constexpr size_t NS_IDX = 4;
} // namespace

namespace optiling {
static ge::graphStatus TilingForGroupPoints(gert::TilingContext* context)
{
    GroupPointsTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // get platformInfo
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfo);
    auto inputInf = context->GetInputDesc(0);
    if (inputInf == nullptr) {
        return ge::GRAPH_FAILED;
    }

    static uint32_t aivCoreNum = ascendplatformInfo.GetCoreNumAiv();
    if (aivCoreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSize;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t availableUbSize = ubSize - RPC_WORKSIZE;

    // get attrs
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
    auto batchSize = getAttr(B_IDX);
    auto cSize = getAttr(C_IDX);
    auto nSize = getAttr(N_IDX);
    auto npoints = getAttr(NP_IDX);
    auto nsample = getAttr(NS_IDX);
    auto dtype = inputInf->GetDataType();
    uint32_t dtypeSize = (dtype == ge::DT_FLOAT) ? SIZE_OF_FP32 : SIZE_OF_FP16;
    uint32_t elemAligned32B = BLOCK_SIZE / dtypeSize;
    uint32_t totalTaskNum = batchSize * npoints * nsample;

    uint32_t coreTaskNum = DivCeil(totalTaskNum, aivCoreNum);
    coreTaskNum = CeilAlign(coreTaskNum, MIN_CORE_TASK);
    if (coreTaskNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t useCoreNum = DivCeil(totalTaskNum, coreTaskNum);
    uint32_t lastCoreTaskNum = (totalTaskNum % coreTaskNum == 0) ? coreTaskNum : (totalTaskNum % coreTaskNum);

    uint32_t cAligned = CeilAlign(static_cast<uint32_t>(cSize), elemAligned32B);
    uint64_t singleTaskSize = cAligned * dtypeSize + SIZE_OF_INT32;
    uint32_t maxUbTaskNum = FloorAlign(std::min(MAX_COPY_BLOCK_COUNT, DivFloor(availableUbSize, singleTaskSize)),
        static_cast<uint64_t>(UB_TASK_BLOCK));
    if (maxUbTaskNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t lastCoreTailAligned = CeilAlign(lastCoreTaskNum % maxUbTaskNum, UB_TASK_BLOCK);

    context->SetBlockDim(useCoreNum);
    tiling.set_useCoreNum(useCoreNum);
    tiling.set_batchSize(batchSize);
    tiling.set_cSize(cSize);
    tiling.set_nSize(nSize);
    tiling.set_npoints(npoints);
    tiling.set_nsample(nsample);
    tiling.set_cAligned(cAligned);
    tiling.set_maxUbTaskNum(maxUbTaskNum);
    tiling.set_coreTaskNum(coreTaskNum);
    tiling.set_lastCoreTaskNum(lastCoreTaskNum);
    tiling.set_mainCoreLoop(coreTaskNum / maxUbTaskNum);
    tiling.set_mainCoreTail(coreTaskNum % maxUbTaskNum);
    tiling.set_lastCoreLoop(lastCoreTaskNum / maxUbTaskNum);
    tiling.set_lastCoreTail(lastCoreTaskNum % maxUbTaskNum);
    tiling.set_lastCoreTailAligned(lastCoreTailAligned);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling


namespace ge {
static ge::graphStatus InferShapeForGroupPoints(gert::InferShapeContext* context)
{
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
    auto batchSize = getAttr(B_IDX);
    auto npoints = getAttr(NP_IDX);
    auto nsample = getAttr(NS_IDX);
    auto cSize = getAttr(C_IDX);

    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outShape = {batchSize * npoints * nsample, cSize};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForGroupPoints(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    if (inputDataType == DT_UNDEFINED) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
} // namespace ge


namespace ops {
class GroupPoints : public OpDef {
public:
    explicit GroupPoints(const char* name) : OpDef(name)
    {
        this->Input("points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("b").AttrType(REQUIRED).Int();
        this->Attr("c").AttrType(REQUIRED).Int();
        this->Attr("n").AttrType(REQUIRED).Int();
        this->Attr("npoints").AttrType(REQUIRED).Int();
        this->Attr("nsample").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShapeForGroupPoints).SetInferDataType(ge::InferDataTypeForGroupPoints);

        this->AICore().SetTiling(optiling::TilingForGroupPoints);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GroupPoints);
} // namespace ops