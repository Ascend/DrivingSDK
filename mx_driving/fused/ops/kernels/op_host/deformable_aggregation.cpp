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
#include "deformable_aggregation_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {


constexpr uint32_t SINGLE = 1;
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t SIZE_OF_FP32 = 4;
constexpr uint32_t BATCH_SIZE_IDX = 0;
constexpr uint32_t FEAT_IDX = 1;
constexpr uint32_t EMBEDS_IDX = 2;
constexpr uint32_t ANCHORS_IDX = 3;
constexpr uint32_t PTS_IDX = 4;
constexpr uint32_t CAMS_IDX = 5;
constexpr uint32_t SCALE_IDX = 6;
constexpr uint32_t GROUPS_IDX = 7;


} // namespace
namespace optiling {

static ge::graphStatus TilingForDeformableAggregation(gert::TilingContext* context)
{
    DeformableAggregationTilingData tiling;
    if (context == nullptr) {
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

    auto dtype = context->GetInputDesc(0)->GetDataType();

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

    auto bs = getAttr(BATCH_SIZE_IDX);
    auto numFeats = getAttr(FEAT_IDX);
    auto numEmbeds = getAttr(EMBEDS_IDX);
    auto numAnchor = getAttr(ANCHORS_IDX);
    auto numPoints = getAttr(PTS_IDX);
    auto numCams = getAttr(CAMS_IDX);
    auto numScale = getAttr(SCALE_IDX);
    auto numGroups = getAttr(GROUPS_IDX);

    uint32_t alignNum = BYTE_BLOCK / SIZE_OF_FP32;
    uint32_t cAligned = (numEmbeds + alignNum - 1) / alignNum * alignNum;
    uint32_t singleAligned = (SINGLE + alignNum - 1) / alignNum * alignNum;

    uint32_t average = bs * numAnchor * numPoints * numCams * numScale / coreNum;
    uint32_t taskLast = bs * numAnchor * numPoints * numCams * numScale % coreNum;
    uint32_t groupAligned = (numGroups + alignNum - 1) / alignNum * alignNum;
    uint32_t usedCoreNum = coreNum;
    if (average == 0) {
        usedCoreNum = taskLast;
    }

    context->SetBlockDim(usedCoreNum);

    tiling.set_bs(bs);
    tiling.set_numFeats(numFeats);
    tiling.set_numEmbeds(numEmbeds);
    tiling.set_numAnchor(numAnchor);
    tiling.set_numPoints(numPoints);
    tiling.set_numCams(numCams);
    tiling.set_numScale(numScale);
    tiling.set_numGroups(numGroups);
    tiling.set_cAligned(cAligned);
    tiling.set_singleAligned(singleAligned);
    tiling.set_average(average);
    tiling.set_taskLast(taskLast);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_groupAligned(groupAligned);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling


namespace ge {
static ge::graphStatus InferShapeForDeformableAggregation(gert::InferShapeContext* context)
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
    auto bs = getAttr(BATCH_SIZE_IDX);
    auto anchor = getAttr(ANCHORS_IDX);
    auto c = getAttr(EMBEDS_IDX);

    gert::Shape* outShape = context->GetOutputShape(0);
    *outShape = {bs, anchor, c};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForDeformableAggregation(gert::InferDataTypeContext* context)
{
    const ge::DataType value_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, value_dtype);
    return GRAPH_SUCCESS;
}

} // namespace ge


namespace ops {
class DeformableAggregation : public OpDef {
public:
    explicit DeformableAggregation(const char* name) : OpDef(name)
    {
        this->Input("mc_ms_feat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("spatial_shape")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scale_start_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sampling_location")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("batch_size").AttrType(REQUIRED).Int();
        this->Attr("num_feat").AttrType(REQUIRED).Int();
        this->Attr("num_embeds").AttrType(REQUIRED).Int();
        this->Attr("num_anchors").AttrType(REQUIRED).Int();
        this->Attr("num_pts").AttrType(REQUIRED).Int();
        this->Attr("num_cams").AttrType(REQUIRED).Int();
        this->Attr("num_scale").AttrType(REQUIRED).Int();
        this->Attr("num_groups").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShapeForDeformableAggregation)
            .SetInferDataType(ge::InferDataTypeForDeformableAggregation);

        this->AICore().SetTiling(optiling::TilingForDeformableAggregation);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(DeformableAggregation);
} // namespace ops