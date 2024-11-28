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
#include "deformable_aggregation_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace ge;
using namespace std;

namespace {

constexpr uint32_t SINGLE = 1;
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t SIZE_OF_FP32 = 4;

const uint32_t INPUT_FEAT = 0;
const uint32_t INPUT_SPATIAL_SHAPE = 1;
const uint32_t INPUT_SAMPLING_LOCATION = 3;
const uint32_t INPUT_WEIGHT = 4;

const uint32_t BATCH_SIZE_DIM = 0;
const uint32_t NUM_FEAT_DIM = 1;
const uint32_t NUM_EMBEDS_DIM = 2;
const uint32_t NUM_CAMS_DIM = 0;
const uint32_t NUM_SCALE_DIM = 1;
const uint32_t NUM_ANCHORS_DIM = 1;
const uint32_t NUM_POINTS_DIM = 2;
const uint32_t NUM_GROUPS_DIM = 5;

} // namespace
namespace optiling {

static ge::graphStatus TilingForDeformableAggregationGrad(gert::TilingContext* context)
{
    DeformableAggregationGradTilingData tiling;

    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto featTensorPtr = context->GetInputTensor(INPUT_FEAT);
    auto spatialShapeTensorPtr = context->GetInputTensor(INPUT_SPATIAL_SHAPE);
    auto samplingLocationTensorPtr = context->GetInputTensor(INPUT_SAMPLING_LOCATION);
    auto WeightTensorPtr = context->GetInputTensor(INPUT_WEIGHT);
    if (featTensorPtr == nullptr || spatialShapeTensorPtr == nullptr || samplingLocationTensorPtr == nullptr || WeightTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto featShape = featTensorPtr->GetStorageShape();
    auto spatialShapeShape = spatialShapeTensorPtr->GetStorageShape();
    auto samplingLocationShape = samplingLocationTensorPtr->GetStorageShape();
    auto weightShape = WeightTensorPtr->GetStorageShape();

    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t batchSize = featShape.GetDim(BATCH_SIZE_DIM);
    uint32_t numFeat = featShape.GetDim(NUM_FEAT_DIM);
    uint32_t numEmbeds = featShape.GetDim(NUM_EMBEDS_DIM);
    uint32_t numCams = spatialShapeShape.GetDim(NUM_CAMS_DIM);
    uint32_t numScale = spatialShapeShape.GetDim(NUM_SCALE_DIM);
    uint32_t numAnchors = samplingLocationShape.GetDim(NUM_ANCHORS_DIM);
    uint32_t numPoints = samplingLocationShape.GetDim(NUM_POINTS_DIM);
    uint32_t numGroups = weightShape.GetDim(NUM_GROUPS_DIM);
    if (numGroups == 0) {
        return -1;
    }
    
    uint32_t alignNum = BYTE_BLOCK / SIZE_OF_FP32;

    uint32_t cAligned = (numEmbeds / numGroups + alignNum - 1) / alignNum * alignNum;
    uint32_t singleAligned = (SINGLE + alignNum - 1) / alignNum * alignNum;

    uint32_t average = batchSize * numAnchors * numPoints * numCams * numScale / coreNum;
    uint32_t taskLast = batchSize * numAnchors * numPoints * numCams * numScale % coreNum;

    uint32_t usedCoreNum = coreNum;
    if (average == 0) {
        usedCoreNum = taskLast;
    }
    context->SetBlockDim(usedCoreNum);

    tiling.set_batchSize(batchSize);
    tiling.set_numCams(numCams);
    tiling.set_numFeat(numFeat);
    tiling.set_numEmbeds(numEmbeds);
    tiling.set_numScale(numScale);
    tiling.set_numAnchors(numAnchors);
    tiling.set_numPoints(numPoints);
    tiling.set_numGroups(numGroups);
    
    tiling.set_cAligned(cAligned);
    tiling.set_singleAligned(singleAligned);
    
    tiling.set_average(average);
    tiling.set_taskLast(taskLast);
    tiling.set_usedCoreNum(usedCoreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling


namespace ge {
static ge::graphStatus InferShapeForDeformableAggregationGrad(gert::InferShapeContext* context)
{
    const gert::Shape* featShape = context->GetInputShape(INPUT_FEAT);
    if (featShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* samplingLocationShape = context->GetInputShape(INPUT_SAMPLING_LOCATION);
    if (samplingLocationShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* weightShape = context->GetInputShape(INPUT_WEIGHT);
    if (weightShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* grad_mc_ms_feat_shape = context->GetOutputShape(0);
    gert::Shape* grad_sampling_location_shape = context->GetOutputShape(1);
    gert::Shape* grad_weight_shape = context->GetOutputShape(2);
    if ((grad_mc_ms_feat_shape == nullptr) || (grad_sampling_location_shape == nullptr) || (grad_weight_shape == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    *grad_mc_ms_feat_shape = *featShape;
    *grad_sampling_location_shape = *samplingLocationShape;
    *grad_weight_shape = *weightShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForDeformableAggregationGrad(gert::InferDataTypeContext* context)
{
    const ge::DataType value_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, value_dtype);
    return GRAPH_SUCCESS;
}

} // namespace ge


namespace ops {
class DeformableAggregationGrad : public OpDef {
public:
    explicit DeformableAggregationGrad(const char* name) : OpDef(name)
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
        this->Input("grad_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
       this->Output("grad_mc_ms_feat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_sampling_location")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForDeformableAggregationGrad)
            .SetInferDataType(ge::InferDataTypeForDeformableAggregationGrad);

        this->AICore().SetTiling(optiling::TilingForDeformableAggregationGrad);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(DeformableAggregationGrad);
} // namespace ops
