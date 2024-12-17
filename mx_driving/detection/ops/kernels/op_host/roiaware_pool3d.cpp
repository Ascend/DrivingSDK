/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "roiaware_pool3d_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
using namespace ge;
using namespace std;

namespace {
const uint32_t INPUT_ROIS = 0;
const uint32_t INPUT_PTS = 1;
const uint32_t INPUT_PTS_FEATRUE = 2;

const uint32_t INPUT_MODE = 0;
const uint32_t MAX_PTS_PER_VOXEL = 1;
const uint32_t OUT_X = 2;
const uint32_t OUT_Y = 3;
const uint32_t OUT_Z = 4;

const uint32_t BOX_SIZE_DIM = 0;
const uint32_t PTS_SIZE_DIM = 0;
const uint32_t CHANNEL_DIM = 1;

const uint32_t OUPUT_ARGMAX = 0;
const uint32_t PTS_IDX_OF_VOXEL = 1;
const uint32_t POOL_FEATURES = 2;

const uint32_t WORKSAPCE_16MBYTE_SIZE = 16 * 1024 * 1024;
} // namespace

namespace optiling {
static ge::graphStatus TilingFuncForRoiawarePool3d(gert::TilingContext* context)
{
    RoiawarePool3dTilingData tiling;

    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto roisTensorPtr = context->GetInputTensor(INPUT_ROIS);
    auto ptsTensorPtr = context->GetInputTensor(INPUT_PTS);
    auto ptsFeaturePtr = context->GetInputTensor(INPUT_PTS_FEATRUE);
    if (roisTensorPtr == nullptr || ptsTensorPtr == nullptr || ptsFeaturePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    uint64_t ub_total_size;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_total_size);
    uint32_t coreNum = ascendplatformInfo.GetCoreNumAiv();
    context->SetBlockDim(coreNum);
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto roiShape = roisTensorPtr->GetStorageShape();
    auto ptsShape = ptsTensorPtr->GetStorageShape();
    auto ptsFeatureShape = ptsFeaturePtr->GetStorageShape();

    uint32_t boxNum = roiShape.GetDim(BOX_SIZE_DIM);
    uint32_t ptsNum = ptsShape.GetDim(PTS_SIZE_DIM);
    uint32_t channelNum = ptsFeatureShape.GetDim(CHANNEL_DIM);

    uint32_t mode = *(attrs->GetAttrPointer<uint32_t>(INPUT_MODE));
    uint32_t maxPtsPerVoxel = *(attrs->GetAttrPointer<uint32_t>(MAX_PTS_PER_VOXEL));
    uint32_t outx = *(attrs->GetAttrPointer<uint32_t>(OUT_X));
    uint32_t outy = *(attrs->GetAttrPointer<uint32_t>(OUT_Y));
    uint32_t outz = *(attrs->GetAttrPointer<uint32_t>(OUT_Z));

    uint32_t coreBoxNums = boxNum / coreNum;
    uint32_t coreBoxTail = boxNum % coreNum;

    tiling.set_coreBoxNums(coreBoxNums);
    tiling.set_coreBoxTail(coreBoxTail);
    tiling.set_boxNum(boxNum);
    tiling.set_ptsNum(ptsNum);
    tiling.set_channelNum(channelNum);
    tiling.set_maxPtsPerVoxel(maxPtsPerVoxel);
    tiling.set_outx(outx);
    tiling.set_outy(outy);
    tiling.set_outz(outz);
    tiling.set_mode(mode);
    tiling.set_coreNum(coreNum);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = WORKSAPCE_16MBYTE_SIZE;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShapeForRoiawarePool3d(gert::InferShapeContext* context)
{
    const gert::Shape* box_shape = context->GetInputShape(INPUT_ROIS);
    const gert::Shape* feature_shape = context->GetInputShape(INPUT_PTS_FEATRUE);
    if (box_shape == nullptr || feature_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t box_num = box_shape->GetDim(BOX_SIZE_DIM);
    uint32_t channel_num = feature_shape->GetDim(CHANNEL_DIM);
    auto attrsPtr = context->GetAttrs();
    uint32_t maxPtsPerVoxel = *(attrsPtr->GetAttrPointer<uint32_t>(MAX_PTS_PER_VOXEL));
    uint32_t outx = *(attrsPtr->GetAttrPointer<uint32_t>(OUT_X));
    uint32_t outy = *(attrsPtr->GetAttrPointer<uint32_t>(OUT_Y));
    uint32_t outz = *(attrsPtr->GetAttrPointer<uint32_t>(OUT_Z));

    gert::Shape* argmax_shape = context->GetOutputShape(OUPUT_ARGMAX);
    gert::Shape* pts_idx_of_voxel_shape = context->GetOutputShape(PTS_IDX_OF_VOXEL);
    gert::Shape* pooled_features_shape = context->GetOutputShape(POOL_FEATURES);

    *argmax_shape = {box_num, outx, outy, outz, channel_num};
    *pts_idx_of_voxel_shape = {box_num, outx, outy, outz, maxPtsPerVoxel};
    *pooled_features_shape = {box_num, outx, outy, outz, channel_num};
    
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeForRoiawarePool3d(gert::InferDataTypeContext* context)
{
    auto input_dtype = context->GetInputDataType(INPUT_ROIS);
    context->SetOutputDataType(OUPUT_ARGMAX, ge::DT_INT32);
    context->SetOutputDataType(PTS_IDX_OF_VOXEL, ge::DT_INT32);
    context->SetOutputDataType(POOL_FEATURES, input_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class RoiawarePool3d : public OpDef {
public:
    explicit RoiawarePool3d(const char* name) : OpDef(name)
    {
        this->Input("rois")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("pts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("pts_feature")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("mode").AttrType(REQUIRED).Int();
        this->Attr("max_pts_each_voxel").AttrType(REQUIRED).Int();
        this->Attr("outx").AttrType(REQUIRED).Int();
        this->Attr("outy").AttrType(REQUIRED).Int();
        this->Attr("outz").AttrType(REQUIRED).Int();
        this->Output("argmax")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("pts_idx_of_voxels")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("pooled_features")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        
        this->SetInferShape(ge::InferShapeForRoiawarePool3d)
            .SetInferDataType(ge::InferDataTypeForRoiawarePool3d);
        this->AICore().SetTiling(optiling::TilingFuncForRoiawarePool3d);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(RoiawarePool3d);
}