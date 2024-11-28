// Copyright (c) 2024 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
#include "vec_pool_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

using namespace std;

namespace {
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t SIZE_OF_FLOAT = sizeof(float);
constexpr uint32_t SIZE_OF_INT = sizeof(int32_t);
constexpr uint32_t UB_ALIGN_SIZE = 32;
constexpr uint32_t ALIGN_NUM = UB_ALIGN_SIZE / SIZE_OF_FLOAT;
constexpr uint32_t RESERVED_UB_SIZE = 20 * 1024;
constexpr uint32_t IDX_NUM_PER_GROUP = 3;
constexpr uint32_t GROUP_UB_ALIGN_SIZE = 96;
constexpr uint32_t GROUP_SIZE = IDX_NUM_PER_GROUP * SIZE_OF_INT;

constexpr uint32_t GROUPED_IDXS_INPUT_IDX = 2;
constexpr uint32_t GRAD_NEW_FEATURES_INPUT_IDX = 0;
constexpr uint32_t POINT_CNT_OF_GRID_INPUT_IDX = 1;

constexpr uint32_t ATTR_N_IDX = 0;
constexpr uint32_t ATTR_CIN_IDX = 1;

constexpr uint32_t INPUT_DIM = 2;

int32_t GetCeilInt(int32_t value1, int32_t value2)
{
    if (value2 == 0) {
        return value1;
    }
    return (value1 + value2 - 1) / value2;
}
}

namespace optiling {
static ge::graphStatus VecPoolGradTilingFunc(gert::TilingContext *context)
{
    VecPoolGradTilingData tiling;
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();

    auto groupedIdxsShapePtr = context->GetInputTensor(GROUPED_IDXS_INPUT_IDX);
    auto gradNewFeaturesShapePtr = context->GetInputTensor(GRAD_NEW_FEATURES_INPUT_IDX);
    auto pointCntOfGridShapePtr = context->GetInputTensor(POINT_CNT_OF_GRID_INPUT_IDX);
    if (groupedIdxsShapePtr == nullptr || gradNewFeaturesShapePtr == nullptr || pointCntOfGridShapePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto groupedIdxsShape = groupedIdxsShapePtr->GetStorageShape();
    auto gradNewFeaturesShape = gradNewFeaturesShapePtr->GetStorageShape();
    auto pointCntOfGridShape = pointCntOfGridShapePtr->GetStorageShape();
    if (groupedIdxsShape.GetDimNum() != INPUT_DIM ||
        gradNewFeaturesShape.GetDimNum() != INPUT_DIM ||
        pointCntOfGridShape.GetDimNum() != INPUT_DIM) {
        return ge::GRAPH_FAILED;
    }
    uint32_t numMaxSumPoints = groupedIdxsShape.GetDim(0);
    uint32_t formerCoreGroups = GetCeilInt(numMaxSumPoints, coreNum);
    uint32_t usedCoreNum = GetCeilInt(numMaxSumPoints, formerCoreGroups);
    uint32_t lastCoreGroups = numMaxSumPoints - (usedCoreNum - 1) * formerCoreGroups;
    if (formerCoreGroups == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t formerCoreData = formerCoreGroups * GROUP_SIZE;
    uint32_t lastCoreData = lastCoreGroups * GROUP_SIZE;

    uint32_t cOut = gradNewFeaturesShape.GetDim(1);
    uint32_t numTotalGrids = pointCntOfGridShape.GetDim(1);
    uint32_t numCEachGrid = cOut / numTotalGrids;
    uint32_t gradUBEleNum = GetCeilInt(numCEachGrid, ALIGN_NUM) * ALIGN_NUM;
    uint64_t pointCntOfGridUbSize = UB_ALIGN_SIZE * BUFFER_NUM;
    uint64_t gradNewFeaturesUbSize = GetCeilInt(gradUBEleNum * SIZE_OF_FLOAT, UB_ALIGN_SIZE) * UB_ALIGN_SIZE * BUFFER_NUM;
    uint64_t gradSupportFeaturesUbSize = gradNewFeaturesUbSize;
    uint64_t usedUbSize = pointCntOfGridUbSize + gradNewFeaturesUbSize + gradSupportFeaturesUbSize;

    uint64_t availableUbSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, availableUbSize);
    availableUbSize = availableUbSize - usedUbSize - RESERVED_UB_SIZE;
    availableUbSize = availableUbSize / BUFFER_NUM / GROUP_UB_ALIGN_SIZE * GROUP_UB_ALIGN_SIZE;
    uint32_t formerTilingNum = GetCeilInt(formerCoreData, availableUbSize);
    uint32_t mainGroups = availableUbSize / GROUP_SIZE;
    uint32_t copyTail = formerCoreData % availableUbSize;
    uint32_t formerTailGroups = copyTail / GROUP_SIZE;
    uint32_t lastTilingNum = GetCeilInt(lastCoreData, availableUbSize);
    uint32_t lastCopyTail = lastCoreData % availableUbSize;
    uint32_t lastTailGroups = lastCopyTail / GROUP_SIZE;

    context->SetBlockDim(usedCoreNum);
    uint32_t m = gradNewFeaturesShape.GetDim(0);
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
    int32_t n = getAttr(ATTR_N_IDX);
    int32_t cIn = getAttr(ATTR_CIN_IDX);
    int32_t repeatTimes = cIn / numCEachGrid;
    int32_t tail = cIn % numCEachGrid;
    int32_t mainCopySize = mainGroups * IDX_NUM_PER_GROUP * SIZE_OF_INT;
    int32_t formerCoreTailCopySize = formerTailGroups * IDX_NUM_PER_GROUP * SIZE_OF_INT;
    int32_t lastCoreTailCopySize = lastTailGroups * IDX_NUM_PER_GROUP * SIZE_OF_INT;
    tiling.set_formerCoreGroups(formerCoreGroups);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_availableUbSize(availableUbSize);
    tiling.set_mainGroups(mainGroups);
    tiling.set_copyLoop(formerTilingNum - 1);
    tiling.set_copyTail(copyTail);
    tiling.set_formerTailGroups(formerTailGroups);
    tiling.set_lastCopyLoop(lastTilingNum - 1);
    tiling.set_lastCopyTail(lastCopyTail);
    tiling.set_lastTailGroups(lastTailGroups);
    tiling.set_m(m);
    tiling.set_cOut(cOut);
    tiling.set_numTotalGrids(numTotalGrids);
    tiling.set_numCEachGrid(numCEachGrid);
    tiling.set_gradUBEleNum(gradUBEleNum);
    tiling.set_numMaxSumPoints(numMaxSumPoints);
    tiling.set_n(n);
    tiling.set_cIn(cIn);
    tiling.set_repeatTimes(repeatTimes);
    tiling.set_tail(tail);
    tiling.set_mainCopySize(mainCopySize);
    tiling.set_formerCoreTailCopySize(formerCoreTailCopySize);
    tiling.set_lastCoreTailCopySize(lastCoreTailCopySize);

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
    int32_t n = getAttr(0);
    int32_t cIn = getAttr(1);

    gert::Shape* gradSupportFeaturesShape = context->GetOutputShape(0);
    if (gradSupportFeaturesShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gradSupportFeaturesShape->AppendDim(n);
    gradSupportFeaturesShape->AppendDim(cIn);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForVecPoolGrad(gert::InferDataTypeContext* context)
{
    const ge::DataType value_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, value_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class VecPoolGrad : public OpDef {
public:
    explicit VecPoolGrad(const char* name) : OpDef(name)
    {
        this->Input("grad_new_features")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("point_cnt_of_grid")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("grouped_idxs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_support_features")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("n").Int();
        this->Attr("num_c_in").Int();

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataTypeForVecPoolGrad);;

        this->AICore()
            .SetTiling(optiling::VecPoolGradTilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(VecPoolGrad);
}
